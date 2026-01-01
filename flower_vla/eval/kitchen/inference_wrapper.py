import functools
from typing import Optional
import numpy as np
import torch
from flower_vla.dataset.oxe.transforms import generate_policy_prompt, get_action_space_index
from flower_vla.dataset.utils.frequency_mapping import DATASET_FREQUENCY_MAP
from flower_vla.eval.simpler.flower_inference_wrapper import UhaInference as SimplerUhaInference
from PIL import Image

KIT_IRL_REAL_KITCHEN_DATASET_INDICES = [
    1,  # kit_irl_real_kitchen_delta_des_joint_euler
    2,  # kit_irl_real_kitchen_vis_delta_des_joint_euler
    3,  # kit_irl_real_kitchen_lang
    4,  # kit_irl_real_kitchen_vis
]
PNP_SCORE_DATASET_INDICES = [
    73,  # pnp_score_eef
    74,  # pnp_score_joint
]

class UhaInference(SimplerUhaInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_history =[]
        # print("[Kitchen UhaInference] image_size =", self.image_size) # torch.Size([1, 1, 3, 224, 224])

        # kitchen uses both primary and secondary
        # self.agent.agent.img_modalities = ["image_primary", "image_secondary"]
        self.agent.agent.img_modalities = ["image_primary", "image_wrist"]
        # self.agent.agent.img_modalities = ["image_primary"]

        #----joint position----#
        # new_view tomato_pnp
        # self.max_values = torch.tensor([-0.18627665460109707, 0.6912829279899597, 0.0031117144855670634, -1.5747924602031707, 0.027757519856095314, 2.7066869735717773, 2.2141610860824583, 1.0])
        # self.min_values = torch.tensor([-1.428738555908203, -0.7126483064889908, -0.008124510487541557, -2.780069522857666, -0.018890886697918177, 1.6770271003246306, 0.9807969146966934, 0.0])
        # new_view tomato_pnp fixed point
        # self.max_values = torch.tensor([-0.1874053803086278, 0.4903961545228958, 0.0006624547298997704, -1.9006540155410767, 0.022062694393098353, 2.7405100297927865, 2.2149619960784914, 1.0])
        # self.min_values = torch.tensor([-1.1405200958251953, -0.7216042470932007, -0.007208807291463017, -2.693549175262451, -0.010610386729240417, 1.676675169467926, 1.244762716293335, 0.0])
        # pnp_flower
        self.max_values = torch.tensor([-0.13768272697925568, 0.6396570086479182, 0.002858846168965098, -1.671217679977417, 0.036994095891714096, 2.7766952991485594, 2.2399791717529296, 1.0])
        self.min_values = torch.tensor([-1.717871642112732, -0.9164057850837708, -0.008966750465333462, -2.962847089767456, -0.018864415213465692, 1.6743356704711914, 0.6593724370002747, 0.0])

        # Language Instruction
        self.format_instruction = functools.partial(
                generate_policy_prompt,
                robot_name="Franka Panda",
                action_space="joint position",
                # action_space="eef velocity",
                num_arms="1",
                prompt_style='minimal'
            )
        
        # Action processing
        self.action_space_index = torch.tensor([get_action_space_index(robot_type='JOINT_POS', num_arms=1, control_mode='position', return_tensor=False)])
        # self.action_space_index = torch.tensor([get_action_space_index(robot_type='JOINT_POS', num_arms=1, control_mode='velocity', return_tensor=False)])
        # self.action_space_index = torch.tensor([get_action_space_index(robot_type='EEF_POS', num_arms=1, control_mode='velocity', return_tensor=False)])

        # self.frequency = torch.tensor([DATASET_FREQUENCY_MAP[KIT_IRL_REAL_KITCHEN_DATASET_INDICES[0]]])
        self.frequency = torch.tensor([DATASET_FREQUENCY_MAP[PNP_SCORE_DATASET_INDICES[1]]])  # 1

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        img: [H, W, 3], uint8
        1. 用父类的 _resize_image 按 self.image_size resize 成正方形
        2. 变成 [1, 1, 3, H, W] 的 torch.uint8 tensor，并放到 self.device
        """
        assert img.ndim == 3 and img.shape[2] == 3, f"expected HWC image, got shape {img.shape}"

        # 调用父类的 _resize_image
        img_resized = self._resize_image(img)   # 结果是 [self.image_size, self.image_size, 3]，例如 224x224
        # img_resized = img
        tensor = (
            torch.as_tensor(img_resized, dtype=torch.uint8)
            .permute(2, 0, 1)      # [3, H, W]
            .unsqueeze(0)          # [1, 3, H, W]
            .unsqueeze(0)          # [1, 1, 3, H, W]
            .to(self.device)
        )
        return tensor
    
    def save_tensor_image(self, img: torch.Tensor, path: str):
        """
        img: torch.Tensor [1, 1, 3, H, W] or [3, H, W]
        """
        img = img.squeeze(0).squeeze(0)   # [3, H, W]
        img = img.permute(1, 2, 0)         # [H, W, 3]
        img = img.cpu().numpy()

        Image.fromarray(img).save(path)

        

    def step(self, primary_image: np.ndarray, wrist_image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            primary_image: np.ndarray of shape (H, W, 3), uint8
            wrist_image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_actions: np.ndarray; raw policy action output
        """
        task_description = self.format_instruction(task_description)
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)
                self.agent.agent.reset()
                self.act_chunk_deque.clear()

        if self.ensemble_strategy == 'false' or self.ensemble_strategy is None:
            self.agent.agent.return_act_chunk = False
        else:
            self.agent.agent.return_act_chunk = True

        assert primary_image.dtype == np.uint8
        assert wrist_image.dtype == np.uint8
        
        primary_image = self._preprocess_image(primary_image)  # [1, 1, 3, H, W], square
        wrist_image = self._preprocess_image(wrist_image)  # [1, 1, 3, H, W], square
        # primary_image = torch.as_tensor(primary_image, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
        # wrist_image = torch.as_tensor(wrist_image, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
        self.save_tensor_image(primary_image, "/home/yuan/flower_vla_pret/outputs/primary_image.png")
        self.save_tensor_image(wrist_image, "/home/yuan/flower_vla_pret/outputs/wrist_image.png")
        print(primary_image.shape)
        input_observation = {
            "observation": {
                "image_primary": primary_image,
                "image_wrist": wrist_image,
                "pad_mask_dict": {
                    "image_primary": torch.ones(1,1).bool(),
                    "image_wrist": torch.ones(1,1).bool(),
                },
            },
            "task": {
                "language_instruction": self.task_description_embedding,
                "frequency": self.frequency,
                "action_space_index": self.action_space_index,
            }
        }
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                unscaled_raw_actions = self.agent(input_observation).cpu() # (action_dim)

        # next do custom ensemble strategy depending on the environment
         # Apply ensemble strategy if enabled (before rescaling)
        if self.ensemble_strategy == "act":
            # Convert to numpy for ensemble processing
            act_chunk = unscaled_raw_actions[..., :self.action_index.get_action_dim(self.action_space_index)]
            single_action = self.ensemble_action(act_chunk)
            if isinstance(single_action, np.ndarray):
                unscaled_raw_actions = torch.from_numpy(single_action)#.unsqueeze(0)
            elif torch.is_tensor(single_action):
                unscaled_raw_actions = single_action#.unsqueeze(0)
            # unscaled_raw_actions = torch.from_numpy(single_action).unsqueeze(0)
        elif self.ensemble_strategy == "cogact":
            act_chunk = unscaled_raw_actions[..., :self.action_index.get_action_dim(self.action_space_index)]
            single_action = self.cognitive_ensemble_action(act_chunk)
            unscaled_raw_actions = single_action
        else:
            # Convert back to torch tensor
            unscaled_raw_actions = unscaled_raw_actions[..., :self.action_index.get_action_dim(self.action_space_index)]
            unscaled_raw_actions = unscaled_raw_actions

        raw_actions = torch.cat([self.rescale_to_range(unscaled_raw_actions[..., :-1]), unscaled_raw_actions[...,-1:]], dim=-1).detach()

        return raw_actions.detach().cpu().numpy()

    # def step(self, primary_image: np.ndarray, secondary_image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    #     """
    #     Input:
    #         primary_image: np.ndarray of shape (H, W, 3), uint8
    #         secondary_image: np.ndarray of shape (H, W, 3), uint8
    #         task_description: Optional[str], task description; if different from previous task description, policy state is reset
    #     Output:
    #         raw_actions: np.ndarray; raw policy action output
    #     """
    #     task_description = self.format_instruction(task_description)
    #     if task_description is not None:
    #         if task_description != self.task_description:
    #             # task description has changed; reset the policy state
    #             self.reset(task_description)
    #             self.agent.agent.reset()
    #             self.act_chunk_deque.clear()

    #     if self.ensemble_strategy == 'false' or self.ensemble_strategy is None:
    #         self.agent.agent.return_act_chunk = False
    #     else:
    #         self.agent.agent.return_act_chunk = True

    #     assert primary_image.dtype == np.uint8
    #     assert secondary_image.dtype == np.uint8
    #     primary_image = self._preprocess_image(primary_image)      # [1, 1, 3, H, W], square
    #     secondary_image = self._preprocess_image(secondary_image)  # [1, 1, 3, H, W], square
    #     # primary_image = torch.as_tensor(primary_image, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
    #     # secondary_image = torch.as_tensor(secondary_image, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]

    #     input_observation = {
    #         "observation": {
    #             "image_primary": primary_image,
    #             "image_secondary": secondary_image,
    #             "pad_mask_dict": {
    #                 "image_primary": torch.ones(1,1).bool(),
    #                 "image_secondary": torch.ones(1,1).bool(),
    #             },
    #         },
    #         "task": {
    #             "language_instruction": self.task_description_embedding,
    #             "frequency": self.frequency,
    #             "action_space_index": self.action_space_index,
    #         }
    #     }
    #     with torch.no_grad():
    #         with torch.autocast('cuda', dtype=torch.bfloat16):
    #             unscaled_raw_actions = self.agent(input_observation).cpu() # (action_dim)

    #     # next do custom ensemble strategy depending on the environment
    #      # Apply ensemble strategy if enabled (before rescaling)
    #     if self.ensemble_strategy == "act":
    #         # Convert to numpy for ensemble processing
    #         act_chunk = unscaled_raw_actions[..., :self.action_index.get_action_dim(self.action_space_index)]
    #         single_action = self.ensemble_action(act_chunk)
    #         unscaled_raw_actions = torch.from_numpy(single_action).unsqueeze(0)
    #     elif self.ensemble_strategy == "cogact":
    #         act_chunk = unscaled_raw_actions[..., :self.action_index.get_action_dim(self.action_space_index)]
    #         single_action = self.cognitive_ensemble_action(act_chunk)
    #         unscaled_raw_actions = single_action
    #     else:
    #         # Convert back to torch tensor
    #         unscaled_raw_actions = unscaled_raw_actions[..., :self.action_index.get_action_dim(self.action_space_index)]
    #         unscaled_raw_actions = unscaled_raw_actions

    #     raw_actions = torch.cat([self.rescale_to_range(unscaled_raw_actions[..., :-1]), unscaled_raw_actions[...,-1:]], dim=-1).detach()

    #     return raw_actions.detach().cpu().numpy()
    
    def rescale_to_range(self, tensor) -> torch.Tensor:
        max_values = self.max_values.cpu()[..., :tensor.shape[-1]]
        min_values = self.min_values.cpu()[..., :tensor.shape[-1]]
        # Scale the tensor to the new range [new_min, new_max]
        new_min = -torch.ones_like(tensor).cpu()
        new_max = torch.ones_like(tensor).cpu()
        rescaled_tensor = (tensor - new_min) / (new_max - new_min) * (max_values - min_values) + min_values
        return rescaled_tensor
    
        # self.max_values = torch.tensor([ 0.54222493,  0.80284792,  1.0687871 , -0.94879884,  0.06975963,  2.62478238, -0.49846622,  0.07]) # p99 # 1.0
        # self.min_values = torch.tensor([-0.08110087, -0.69933356, -0.17626342, -2.7020722 , -2.43282706,  1.23236138, -2.72828501,  0.00]) # p01 # 0.0
        
        #----joint velocity----#
        # self.max_values = torch.tensor([ 0.0143717 ,  0.02480282,  0.00075005,  0.0207666 ,  0.00344701,  0.02202722,  0.01509143,  1.0 ])  # p99 # 1.0
        # self.min_values = torch.tensor([-0.02148224, -0.02004128, -0.0010196 , -0.01833188, -0.0033629 , -0.01813609, -0.02218748,  0.0 ])  # p01 # 0.0
        #----joint position----#
        # tomato100
        # self.max_values = torch.tensor([-0.2087652488052849, 0.780112682580947, 0.0029720759578049133, -1.362636923789978, 0.09696710109710693, 2.7869317531585693, 2.1888092255592335, 1.0])  # p99 # 1.0
        # self.min_values = torch.tensor([-1.513234008550644, -0.8656652611494065, -0.007898143082857132, -2.9594015526771544, -0.026693327352404594, 1.6928776144981383, 0.8374984538555146, 0.0])  # p01 # 0.0
        # tomato6
        # self.max_values = torch.tensor([-0.14723821759224012, 0.7636982393264771, 0.0032551977690309214, -1.4251147985458414, 0.046224358975887295, 2.769525604248047, 2.2367838287353514, 1.0])  # p99 # 1.0
        # self.min_values = torch.tensor([-1.3032507085800171, -0.7107557249069214, -0.007040463071316481, -2.8252930068969726, -0.010885159261524677, 1.6719314289093017, 1.0717393493652343, 0.0])  # p01 # 0.0
        
        # new_view tomato_pnp
        # self.max_values = torch.tensor([-0.18627665460109707, 0.6912829279899597, 0.0031117144855670634, -1.5747924602031707, 0.027757519856095314, 2.7066869735717773, 2.2141610860824583, 1.0])
        # self.min_values = torch.tensor([-1.428738555908203, -0.7126483064889908, -0.008124510487541557, -2.780069522857666, -0.018890886697918177, 1.6770271003246306, 0.9807969146966934, 0.0])
        # new_view tomatop fixed point
        # self.max_values = torch.tensor([-0.1874053803086278, 0.4903961545228958, 0.0006624547298997704, -1.9006540155410767, 0.022062694393098353, 2.7405100297927865, 2.2149619960784914, 1.0])
        # self.min_values = torch.tensor([-1.1405200958251953, -0.7216042470932007, -0.007208807291463017, -2.693549175262451, -0.010610386729240417, 1.676675169467926, 1.244762716293335, 0.0])
        
        #----eef velocity----#
        # self.max_values = torch.tensor([ 0.00706081,  0.00622422,  0.00717096,  0.00173825,  0.0020998 ,  0.00217522, 1.0])  # p99 # 1.0
        # self.min_values = torch.tensor([-0.0068294 , -0.006758  , -0.00697551, -0.00175675, -0.00203278, -0.00250458, 0.0])  # p01 # 0.0