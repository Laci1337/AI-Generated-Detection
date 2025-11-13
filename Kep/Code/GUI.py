import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

import ClassificationNetwork
import Functions
import App

class GUI():
    def __init__(self) -> None:
        '''
        A Streamlit GUI feluletet megvalosito osztaly konstruktora.
        '''
        st.set_page_config(layout="centered")
        st.title("AI által generált képek detektálása.")
        
        use_cuda_default = torch.cuda.is_available()
        device_choice = st.sidebar.selectbox("Eszköz", ["cuda", "cpu"], index=0 if use_cuda_default else 1)
        self.device = torch.device(device_choice if (device_choice == "cpu" or torch.cuda.is_available()) else "cpu")
        
        st.sidebar.write(f"Aktív eszköz: {self.device}")
        
        self.model = None
        self.image_size = App.image_size
        self.transform = App.transform
        
    def load_model(self, device: torch.device) -> None:
        '''
        Betolti a modellt, ha tudja, egyeb esetben hibat dob.
        '''
        temp_model = ClassificationNetwork.ClassificationNetwork().to(device)
        is_model_loaded = temp_model.load()
        
        if is_model_loaded:
            self.model = temp_model
            self.model.eval()
        else:
            st.error("Model betöltése közben hiba történt.")
            
    def image_uploader(self) -> None:
        '''
        A kep feltolteseert felelo fuggveny.
        '''             
        uploaded_image = st.file_uploader("Tölts fel egy képet", type=["png", "jpg", "jpeg", "webp"])
        
        if uploaded_image is not None:
            pil_img = Image.open(uploaded_image).convert("RGB")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Eredeti")
                st.image(pil_img, use_column_width=True, caption=f"{pil_img.width}×{pil_img.height}")

            # Előfeldolgozás a fenti transformmal
            x = self.transform(pil_img).unsqueeze(0).to(self.device)

            # Inferencia
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                prob = float(probs.squeeze().item())
                prob_percent = prob * 100.0
                #TODO probs adjust to the border
                
                pred_int = int((probs > Functions.border).int().item())

            with c2:
                st.subheader(f"Előfeldolgozott ({self.image_size}×{self.image_size})")
                # vizualizációhoz denormalizáljuk
                x_vis = x[0].detach().cpu()
                x_vis = (x_vis * 0.5) + 0.5
                st.image(transforms.functional.to_pil_image(x_vis.clamp(0, 1)), use_column_width=True)

            st.markdown("---")
            st.subheader("Eredmény")

            st.write(
                f"A kép {prob_percent:.2f}% eséllyel mesterséges intelligencia által generált, {100.0 - prob_percent:.2f}% eséllyel valós"
            )
        else:
            st.info("Tölts fel egy képet a modell futtatásához.")
            
    def run_gui(self):
        self.load_model(self.device)
        self.image_uploader()
        
if __name__ == "__main__":
    image_size = 240
    
    transform = transforms.Compose([
    # 1) középről square crop a rövidebb oldal szerint
    transforms.Lambda(lambda img: transforms.functional.center_crop(img, min(img.size))),
    # 2) átméretezés image_size x image_size-re
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
    
    gui_instance = GUI()
    gui_instance.load_model(gui_instance.device)
    gui_instance.image_uploader()