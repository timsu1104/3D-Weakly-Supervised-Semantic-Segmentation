import os.path as osp
class PseudoDatasetConfig:
    def __init__(self) -> None:

        ###############################
        # Modify this part
        self.folder = "../../dataset/pseudo_images" # where you save your data
        self.cls = "chairs"
        self.text_format = "a chair with white background"

        self.blur_radius = 1
        self.blur_samples = 20
        ###############################

        self.Output_path = osp.join(self.folder, self.cls)

cfg = PseudoDatasetConfig()