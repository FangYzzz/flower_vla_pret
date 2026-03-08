import os
import sys

import cv2
import torch
import time, threading, queue, random
import subprocess
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from datetime import datetime
from pathlib import Path

import pyzed.sl as sl
import base64
from dotenv import load_dotenv
from openai import OpenAI
from groundingdino.util.inference import load_model, load_image, predict, annotate
from loguru import logger

import zerorpc
import asyncio
import websockets
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np


class TaskRewardGeneration:

    def __init__(self, n_tasks: int = 2, max_timesteps: int = 60, max_rounds: int = 1, ws_uri: str = "ws://127.0.0.1:4242"):
        # Configuration
        self.n_tasks = n_tasks
        self.max_timesteps = max_timesteps
        self.current_scene_gdino = None
        self.next_scene_gdino = None

        # Runtime state
        self.round_current = 0
        self.round_next = 1
        self.max_rounds = max_rounds
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.queue: "Queue[tuple[list[str], list[str]]]" = Queue()

        # Hardware / model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "GroundingDINO/weights/groundingdino_swint_ogc.pth",
        ).to(self.device).eval()
        self.zed = None
        self.runtime_parameters = None
        self.image = None
        # self.open_camera()
        self.ws_uri = ws_uri

        # OpenAI client
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Logging
        self.setup_logger()
        logger.info("TaskRewardGeneration initialised (tasks: {}, max_timesteps: {}s)", n_tasks, max_timesteps)

    def setup_logger(self):
        log_dir = f"output_images/{self.timestamp}"
        log_path = os.path.join(log_dir, "log.txt")

        logger.remove()
        logger.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
        logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
        # logger.add(lambda msg: print(msg, end=""), format="{message}")

    # ---------------- WebSocket 通用调用 ----------------
    async def _ws_call(self, method: str, params: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
        """
        低层 WS 调用:
        - 不做 "ok" 判断，只负责发/收 JSON
        - 带超时 & 简单重试
        返回值：完整 resp dict, 例如 {"ok": true, "result": {...}} 或 {"ok": false, "error": "..."}
        """
        attempts = 3
        last_err = None
        for i in range(attempts):
            try:
                async with websockets.connect(self.ws_uri, open_timeout=10, close_timeout=10) as ws:
                    await ws.send(json.dumps({"method": method, "params": params}))
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    resp = json.loads(raw)
                    return resp
            except (asyncio.TimeoutError,
                    websockets.exceptions.ConnectionClosedError,
                    OSError) as e:
                last_err = e
                logger.warning(f"_ws_call {method} attempt {i+1}/{attempts} failed: {e}")
                if i < attempts - 1:
                    # 简单退避
                    await asyncio.sleep(1.0 * (i + 1))
                else:
                    break
        # 重试用尽
        raise RuntimeError(f"_ws_call {method} failed after {attempts} attempts: {last_err}")

    # ---------------- 触发一次评测 ----------------
    def ws_run_eval(self,
                    instruction: str = "pick up the cube and place it in the bowl",
                    external_camera: str = "left",
                    max_timesteps: int = 600,
                    save_videos: bool = False,
                    success_value: float = 1.0) -> Dict[str, Any]:
        params = {
            "instruction": instruction,
            "external_camera": external_camera,
            "max_timesteps": max_timesteps,
            "save_videos": save_videos,
            "success_value": success_value,
        }

        try:
            # 评测可能比较久，timeout 可以设长一点，比如 600s
            resp = asyncio.run(self._ws_call("run_eval", params, timeout=600.0))
        except KeyboardInterrupt:
            # 用户在 A 这边按了 Ctrl+C
            logger.warning("KeyboardInterrupt detected in ws_run_eval, trying to cancel rollout on B...")

            try:
                # 尝试通知 B 停止当前 rollout（需要 B 那边实现 'cancel' 方法）
                asyncio.run(self._ws_call("cancel", {}, timeout=5.0))
                logger.info("Cancel request sent to B successfully.")
            except Exception as e:
                logger.warning(f"Failed to send cancel request to B: {e}")

            # 继续把 KeyboardInterrupt 抛出去，让上层决定是否退出程序
            raise

        # 正常拿到返回结果
        if not resp.get("ok", False):
            err = resp.get("error", "unknown run_eval error")
            raise RuntimeError(f"run_eval failed on B: {err}")

        return resp.get("result", {})

    # ---------------- 从 B 获取一帧 ----------------
    def ws_snapshot(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        从 B 拉取一帧。B 返回 JPEG base64。
        返回:
          {
            "left":  {"bgr": np.ndarray 或 None, "b64": str 或 None},
            "wrist": {"bgr": np.ndarray 或 None, "b64": str 或 None},
          }
        """
        # snapshot 一般比较快，timeout 可以设短一点，比如 30s
        resp = asyncio.run(self._ws_call("snapshot", {"which": "left"}, timeout=30.0))
        if not resp.get("ok", False):
            err = resp.get("error", "unknown snapshot error")
            raise RuntimeError(f"snapshot failed on B: {err}")

        result = resp.get("result", {})
        out: Dict[str, Optional[Dict[str, Any]]] = {"left": None, "wrist": None}

        for key_in, key_out in (("left_image", "left"), ("wrist_image", "wrist")):
            b64jpg = result.get(key_in)
            if not b64jpg:
                out[key_out] = {"bgr": None, "b64": None}
                continue

            # 解码成 BGR 以便本地显示/保存
            arr = np.frombuffer(base64.b64decode(b64jpg), dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.warning(f"cv2.imdecode failed for {key_in}")
                out[key_out] = {"bgr": None, "b64": None}
            else:
                out[key_out] = {"bgr": img_bgr, "b64": b64jpg}
        return out

    def grab_scene(self, round: int, camera: str = "left"):
        """
        返回: (img_rgb, base64_image)
          - img_rgb: HxWx3, RGB (用于你本地后处理/检测/可视化)
          - base64_image: 直接来自 B 的 JPEG base64(可直接发 GPT)
        """
        frames = self.ws_snapshot()
        candidate = frames.get(camera, {})
        if not candidate or candidate.get("bgr") is None:
            raise RuntimeError(f"No valid frame from {camera} camera")

        img_bgr = candidate.get("bgr") # used for viewing
        b64jpg = candidate.get("b64") # used for gpt

        if img_bgr is None or not b64jpg:
            raise RuntimeError("WS snapshot returned no usable frame")

        # Create directory based on timestamp
        save_dir = f"output_images/{self.timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"scene_{round}.jpg")
        cv2.imwrite(save_path, img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # used for gdino

        return img_rgb, b64jpg

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def gdino(self, object, scene, round):
        # IMAGE_PATH = image_path # "weights/dog-3.jpeg"
        TEXT_PROMPT = object # "chair . person . dog ."
        BOX_TRESHOLD = 0.32 # 0.35
        TEXT_TRESHOLD = 0.25 # 0.25

        # image_source, image = load_image(IMAGE_PATH)
        image_source, image = load_image(scene)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        
        save_dir = f"output_images/{self.timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"annotated_image_{round}.jpg")
        cv2.imwrite(save_path, annotated_frame)

        scene_gdino = self.encode_image(save_path)

        return scene_gdino # save_path

    def objects_generation(self, scene):
        prompt_objects = (
            "The table is covered with a black tablecloth and there are two robotic arms nearby.\n"
            "List all the main objects that are **on the black tablecloth**.\n"

            "Only list objects that are **physically on top of the black tablecloth**.\n"
            "Do **not** include:\n"
            "- objects that are not on the tablecloth (e.g., windows, walls, background items).\n"
            "- the tablecloth itself.\n"
            "- the robot arms themselves.\n"
            "- any adjectives such as color, size, material, or quantity.\n"

            "Format your answer as a single line, separating each object with ' . ' and ending with a final ' .'\n"
            "For example: 'cube . tomato . banana . carrot . spoon . cloth . bowl . can .'\n"
        )

        response = self.client.responses.create(
            model="gpt-4.1-mini", # gpt-4.1-mini
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_objects},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{scene}",},
                ],
            }],
        )

        objects_all = response.output_text

        return objects_all

    def task_generation(self, scene, objects_all):
        banned_phrases = [
            "next to", "next", "nearby", "near", "parallel", "close to",
            "on the edge of", "edge", "corner", "according to their orientation",
            "adjacent to", "beside", "alongside", 
            "sides of the table", "separate sides", "both sides"
        ]

        valid_tasks_dict = {}   # taskN: line
        valid_objects_dict = {} # taskN: line
        attempt = 0
        max_attempts = 5  # Avoid infinite loops
        expected_ids = [f"task{i+1}" for i in range(self.n_tasks)]

        while len(valid_tasks_dict) < self.n_tasks and attempt < max_attempts:
            attempt += 1
            remaining = self.n_tasks - len(valid_tasks_dict)

            prompt_task = (
                f"You are a robot with two arms. Based on the scene, generate {remaining} possible tasks the robot could perform.\n"
                f"ONLY use objects from this EXACT list (no more, no less):\n{objects_all}\n"
                "You are NOT allowed to invent new objects.\n"
                "If any object appears in the task that is not in the list, the task is INVALID.\n"
                "Your tasks must follow these allowed action types:\n"
                "- Pick an object from one place and place it into a container.\n"
                "- pick up an object in a container and place it in front of another object.\n"
                "- pick up an object in a container and place it behind another object.\n"
                "- pick up an object in a container and place it to the left of another object.\n"
                "- pick up an object in a container and place it to the right of another object.\n"
                "- pick up an object and place it in front of another object.\n"
                "- pick up an object and place it behind another object.\n"
                "- pick up an object and place it to the left of another object.\n"
                "- pick up an object and place it to the right of another object.\n"

                "Each line must consist of **exactly one intermediate goal**, written as an imperative verb phrase.\n"
                "Use clear imperative phrases. For any pick-and-place action, explicitly state WHAT object is moved and WHERE it is placed.\n"
                "**Strict spatial language rules**:\n"
                "- You must NOT use vague or relative spatial terms.\n"
                "- The following phrases are STRICTLY FORBIDDEN: 'next to', 'nearby', 'parallel', 'close to', 'on the edge of', 'according to their orientation', 'adjacent to', 'beside', 'alongside'.\n"
                "- If any of these appear, the task will be rejected.\n"
                "- You must specify exact destination objects or precise reference items.\n\n"

                "Example format:\n"
                "task1: pick up the tomato and place it into the bowl.\n"
                "task2: pick up the banana and place it into the plate.\n"
                "task3: pick up the cube in the bowl and place it in front of the bowl.\n"
                "task4: pick up the carrot and place it in front of the can.\n"
                
                
                "Now, for each task above, extract the objects involved.\n"
                "Follow these **strict rules**:\n"
                "- You must write exactly one line per task.\n"
                "- Each line must begin with `taskN:` (e.g. `task1:`).\n"
                "- After the colon, list all object names (nouns only), separated by ` . `, and end with a final ` .`\n"
                "- Do **not** include any adjectives (e.g. color, size, quantity).\n"
                "- Do **not** merge the objects from multiple tasks into a single list.\n"
                "- The word 'side', 'task' is not considered an object.\n\n"

                "Correct format example:\n"
                "objects:\n"
                "task1: tomato . bowl .\n"
                "task2: banana . plate .\n"
                "task3: cube . bowl .\n"
                "task4: carrot . can .\n"
            )   

            response = self.client.responses.create(
                model="gpt-4.1-mini", # gpt-4.1-mini
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_task},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{scene}",},
                    ],
                }],
            )

            output = response.output_text
            if "objects:" in output:
                task_block, object_block = output.split("objects:", 1)
            else:
                task_block, object_block = output, ""

            task_lines = [line.strip() for line in task_block.splitlines() if line.lower().startswith("task")]
            object_lines = [line.strip() for line in object_block.splitlines() if line.lower().startswith("task")]

            task_raw = {line.split(":")[0].strip(): line for line in task_lines}
            object_raw = {line.split(":")[0].strip(): line for line in object_lines}

            # Assign missing number
            missing_ids = [tid for tid in expected_ids if tid not in valid_tasks_dict]
            missing_ids_iter = iter(missing_ids)

            for old_id, task_line in task_raw.items():
                task_body = task_line.split(":", 1)[1].strip()

                if any(phrase in task_body.lower() for phrase in banned_phrases):
                    continue

                try:
                    new_task_id = next(missing_ids_iter)
                except StopIteration:
                    break

                valid_tasks_dict[new_task_id] = f"{new_task_id}: {task_body}"

                if old_id in object_raw:
                    obj_body = object_raw[old_id].split(":", 1)[1].strip()
                    valid_objects_dict[new_task_id] = f"{new_task_id}: {obj_body}"

        if len(valid_tasks_dict) < self.n_tasks:
            logger.warning(f"Only {len(valid_tasks_dict)} valid tasks generated after {attempt} attempts (target: {self.n_tasks}).")

        sorted_ids = sorted(valid_tasks_dict.keys(), key=lambda x: int(x[4:]))
        task_output = "\n".join([valid_tasks_dict[tid] for tid in sorted_ids])
        object_output = "\n".join([valid_objects_dict[tid] for tid in sorted_ids if tid in valid_objects_dict])

        tasks = [line.strip().strip() for line in task_output.splitlines() if line.strip().startswith("task")]
        objects = [line.strip().strip() for line in object_output.splitlines() if line.strip().startswith("task")]
        # objects = [line.strip().split(":", 1)[1].strip() for line in object_output.splitlines() if line.strip().startswith("task")]

        return tasks, objects

    def reward_generation(self, current_scene_gdino, next_scene_gdino, current_task):
        prompt_reward = (
            "You are given two images:\n"
            "- The **first image** shows the initial scene **before** the robot starts the task.\n"
            "- The **second image** shows the result **after** the robot attempted the task.\n\n"

            "The robot was instructed to perform the following task:\n"
            f"{current_task}\n\n"

            "Instructions:\n"
            "1. From the human observer's perspective (standing in front of the table, facing the scene), carefully describe the initial position of the key object(s) mentioned in the task.\n"
            # "2. Then describe the final position of the key object(s) in the second image, using clear and precise spatial language.\n"
            "3. Pay close attention to spatial relationships such as left, right, above, below, front, back, inside, outside, parallel, and intersecting.\n"
            "**4. VERY IMPORTANT: If bounding boxes are visible, focus on the exact **box labels** — make sure you refer to the correct object name!**\n"
            "   - Do NOT confuse objects with similar color/shape.\n"
            "   - If labels clearly show containment or alignment, include that in your reasoning.\n"
            # "**5. EVEN MORE IMPORTANT: If segmentation masks are visible along with bounding boxes, rely on BOTH the **label** and the **mask shape/position** to make accurate judgments.**\n"
            # "   - Do NOT confuse objects with similar color/shape. Use the label+mask together.\n"
            # "   - If masks clearly show containment or alignment, include that in your reasoning.\n"
            "6. Compare the final position with the task requirement. "

            "**Respond with only a single digit: `1` if the task was successfully completed, or `0` if it failed.**\n"
            "If the key object(s) are placed exactly as instructed, output `1`.\n"
            "If not (wrong position, ambiguous, or missing), output `0`.\n\n"

            "**Respond with the following format exactly:**\n"
            # "task1: "
            "reward: 1 (or 0)\n"
            "Then on a new line, explain briefly the reason for your answer.\n"
        )   

        response = self.client.responses.create(
            model="gpt-4.1-mini", # gpt-4.1-mini
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_reward},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{current_scene_gdino}",},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{next_scene_gdino}",},
                ],
            }],
        )

        reward = response.output_text

        return reward

    def task_selection(self, tasks, objects):
        selected_task = random.choice(tasks)

        task_id = selected_task.split(":")[0].strip()
        task = next((t.split(":",1)[1].strip()
                            for t in tasks if t.startswith(task_id)), "")
        # object = next((o.split(":",1)[1].strip()
        #                     for o in objects if o.startswith(task_id)), "")

        logger.info(f"Selected {task_id} to execute: {task}")

        # task = "pick up the cube and place it behind the carrot" # pick up the cube and place it into the bowl
        # logger.info(f"Selected follow task to execute: {task}")
        return task # task_id, task, object

    def robot_execution(self, instruction):
        logger.info(">>> Start robot_execution")
        
        result = self.ws_run_eval(
            instruction=instruction,
            external_camera="left",
            max_timesteps=self.max_timesteps,
            save_videos=False,
            success_value=1.0
        )
        # logger.info(f"run_eval result: {result}")
        logger.info("<<< Finished robot_execution")


    def task_thread(self, next_scene, next_objects_all):
        try:
            tasks, objects = self.task_generation(next_scene, next_objects_all)
            self.queue.put((tasks, objects))
        except Exception as e:
            logger.error(f"[Task generation thread] Failed: {e}")

    def reward_thread(self, current_scene_gdino, next_scene_gdino, current_task):
        try:
            reward = self.reward_generation(current_scene_gdino, next_scene_gdino, current_task)
            logger.info(reward)
        except Exception as e:
            logger.error(f"[Reward generation thread] Failed: {e}")

    def main_loop(self):
        try:
            # -------- scene0 -------- 
            current_scene, current_scene_gpt = self.grab_scene(self.round_current)
            current_objects_all = self.objects_generation(current_scene_gpt)
            
            # -------- current_tasks generation --------
            current_tasks, current_objects = self.task_generation(current_scene_gpt, current_objects_all)
            self.queue.put((current_tasks, current_objects))

            # while True:
            while self.round_current <= self.max_rounds:
                # -------- task selection --------
                current_tasks, current_objects = self.queue.get()

                logger.info(f"Objects in the scene: \n{current_objects_all}")
                logger.info("Tasks in the scene:")
                for task in current_tasks:
                    logger.info(f"  {task}")
                logger.info("Objects of each task:")
                for obj in current_objects:
                    logger.info(f"  {obj}")

                # -------- robot execution --------
                current_task = self.task_selection(current_tasks, current_objects)
                self.robot_execution(current_task)

                # -------- scene1,2,3,... -------- 
                next_scene, next_scene_gpt = self.grab_scene(self.round_next)
                next_objects_all = self.objects_generation(next_scene_gpt)

                # -------- gdino detection --------
                if self.round_current == 0:
                    current_scene_gdino = self.gdino(current_objects_all, current_scene, self.round_current)

                next_scene_gdino = self.gdino(next_objects_all, next_scene, self.round_next)
                
                # -------- current_reward & next_tasks generation --------
                fut_reward = self.pool.submit(self.reward_thread, current_scene_gdino, next_scene_gdino, current_task)
                fut_task = self.pool.submit(self.task_thread, next_scene_gpt, next_objects_all)
                wait([fut_reward, fut_task], return_when=ALL_COMPLETED)
                
                logger.info(f"---------------------------------------------------------------- Task{self.round_current} has been completed ----------------------------------------------------------------")
                self.round_current += 1
                self.round_next += 1
                self.current_scene_gdino = next_scene_gdino
                current_scene_gdino = next_scene_gdino
        finally:
            logger.info("Main loop finished.")

if __name__ == '__main__':
    TRG = TaskRewardGeneration(n_tasks=2, max_timesteps=600, max_rounds=9, ws_uri="ws://127.0.0.1:4242") # max_rounds=1 means the robot performs two rounds
    TRG.main_loop()
