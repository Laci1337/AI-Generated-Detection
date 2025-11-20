import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

from ClassificationNetwork import ClassificationNetwork, border0
from App import image_size, transform

class GUI():
    def __init__(self) -> None:
        '''
        A Streamlit GUI feluletet megvalosito osztaly konstruktora.
        '''
        st.set_page_config(layout="centered")
        st.title("AI által generált képek detektálása.")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        self.model = None
        self.image_size = image_size
        self.transform = transform
        
    def load_model(self, device: torch.device) -> None:
        '''
        Betolti a modellt, ha tudja, egyeb esetben hibat dob.
        '''
        temp_model = ClassificationNetwork().to(device)
        is_model_loaded = temp_model.load()
        
        if is_model_loaded:
            self.model = temp_model
            self.model.eval()
        else:
            st.error("Model betöltése közben hiba történt.")
            
    def calculate_probabilities(self, probs: float) -> float:
        '''
        A dontesi hatar alapjan skalazva visszaadja az eselyet annak, hogy egy kep AI generalt
        '''
        if probs > 1.0:
            raise ValueError('A probs parameter nem lehet nagyobb, mint 1!')
        
        if probs > border0:
            rate = (probs - border0) / (1.0 - border0)
            prob = 0.5 + 0.5 * rate
        elif probs < border0:
            rate = (border0 - probs) / border0
            prob = 0.5 - 0.5 * rate
        else:
            prob = 0.5
        
        return prob
            
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

            x = self.transform(pil_img).unsqueeze(0).to(self.device)

            # Inferencia
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                prob = float(probs.squeeze().item())
                adjusted_prob = self.calculate_probabilities(prob)
                adjusted_prob_percent = 100.0 * adjusted_prob
                decision = "Mesterséges intelligenca által generált" if adjusted_prob > 0.5 else "Valós"

            with c2:
                st.subheader(f"Előfeldolgozott ({self.image_size}×{self.image_size})")
                x_vis = x[0].detach().cpu()
                x_vis = (x_vis * 0.5) + 0.5
                st.image(transforms.functional.to_pil_image(x_vis.clamp(0, 1)), use_column_width=True)

            st.markdown("---")
            st.subheader("Eredmény")

            st.write(
                f"A kép {adjusted_prob_percent:.2f}% eséllyel mesterséges intelligencia által generált, {100.0 - adjusted_prob_percent:.2f}% eséllyel valós"
            )
            st.write(f"A modell döntése: {decision}")
        else:
            st.info("Tölts fel egy képet a modell futtatásához.")
            
    def run_gui(self):
        self.load_model(self.device)
        self.image_uploader()
        
if __name__ == "__main__":
    image_size = 240
        
    gui_instance = GUI()
    gui_instance.load_model(gui_instance.device)
    gui_instance.image_uploader()
    
    print(gui_instance.calculate_probabilities(0.95))