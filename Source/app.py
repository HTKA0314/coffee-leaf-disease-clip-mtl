import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import open_clip
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DISEASE_COLS = ["healthy", "miner", "rust", "phoma", "cercospora"]
DISPLAY_DISEASES = ["miner", "rust", "phoma", "cercospora"]

from model import CLIPMultiTaskModel

@gr.cache_resource
def load_model():
    model = CLIPMultiTaskModel().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        thrs = ckpt.get('optimized_thresholds', [0.45, 0.50, 0.40, 0.35])
        print(f"✅ Model Loaded from {MODEL_PATH}!")
    else:
        thrs = [0.45, 0.50, 0.40, 0.35]
        print(f"⚠️ Model not found at {MODEL_PATH}")
    return model, thrs

model, base_thrs = load_model()

treatment_db = {
    "miner": {
        "title": "🐛 SÂU VẼ BÙA (Leucoptera coffeella)",
        "symptoms": "Vết biến dạng ngoằn ngoèo, mỏng, có màu trắng bạc trên bề mặt lá.",
        "tech": (
            "• **Vệ sinh:** Thu gom và tiêu hủy các lá bị hại nặng để diệt nhộng.\n"
            "• **Thiên địch:** Hạn chế phun thuốc phổ rộng để bảo vệ kiến vàng và các loài ong ký sinh họ Braconidae.\n"
            "• **Canh tác:** Cắt tỉa các chồi vượt, cành sát đất để vườn cây thông thoáng."
        ),
        "chemical": (
            "• **Hoạt chất:** Abamectin (Reasgant), Imidacloprid (Confidor).\n"
            "• **Thời điểm:** Phun khi cây bắt đầu ra lá non hoặc khi tỷ lệ lá bị hại > 10%.\n"
            "• **Lưu ý:** Phun ướt đều cả hai mặt lá."
        )
    },
    "rust": {
        "title": "🍂 BỆNH RỈ SẮT (Hemileia vastatrix)",
        "symptoms": "Các bào tử màu cam rực rỡ mặt dưới lá, sau đó vết bệnh cháy khô và rụng lá hàng loạt.",
        "tech": (
            "• **Chọn giống:** Sử dụng các giống kháng bệnh như TR4, TR9 hoặc các dòng Catimor.\n"
            "• **Cắt tỉa:** Loại bỏ cành tăm, cành vô hiệu để giảm độ ẩm trong tán cây.\n"
            "• **Dinh dưỡng:** Tránh bón thừa đạm (N), tăng cường Kali (K2O) để tăng sức đề kháng."
        ),
        "chemical": (
            "• **Hoạt chất:** Cyproconazole (Anvil 5SC), Hexaconazole, hoặc Tebuconazole.\n"
            "• **Thời điểm:** Phun phòng vào đầu mùa mưa hoặc khi vừa chớm xuất hiện vết bệnh."
        )
    },
    "phoma": {
        "title": "🌑 BỆNH PHOMA (Phoma spp.)",
        "symptoms": "Vết bệnh hình tròn màu nâu đen, thường bắt đầu từ mép lá hoặc chóp lá.",
        "tech": (
            "• **Thoát nước:** Đảm bảo vườn không bị úng cục bộ, đặc biệt là trong mùa mưa kéo dài.\n"
            "• **Vệ sinh:** Cắt bỏ các cành bị khô và hoại tử do nấm tấn công.\n"
            "• **Khoảng cách:** Đảm bảo mật độ trồng hợp lý, không quá dày."
        ),
        "chemical": (
            "• **Hoạt chất:** Copper oxychloride (COC 85), Carbendazim hoặc Mancozeb.\n"
            "• **Cách dùng:** Phun kỹ vào các kẽ cành và mặt dưới lá nơi nấm dễ tích tụ."
        )
    },
    "cercospora": {
        "title": "👁️ BỆNH ĐỐM MẮT CUA (Cercospora coffeicola)",
        "symptoms": "Đốm tròn nhỏ, tâm màu xám trắng, viền ngoài màu nâu đỏ trông như mắt cua.",
        "tech": (
            "• **Cây che bóng:** Điều chỉnh độ che phủ hợp lý (khoảng 30%), tránh để nắng gắt chiếu trực tiếp quá nhiều.\n"
            "• **Phân bón:** Bổ sung phân hữu cơ và các vi lượng (Zn, Mg, Boron) giúp lá dày, khỏe.\n"
            "• **Tưới nước:** Tưới nước đều đặn, tránh để cây bị stress hạn hán."
        ),
        "chemical": (
            "• **Hoạt chất:** Chlorothalonil (Daconil), Azoxystrobin + Difenoconazole (Amistar Top).\n"
            "• **Lưu ý:** Thường xuất hiện trên vườn chăm sóc kém hoặc cây bị kiệt sức sau thu hoạch."
        )
    }
}

def get_recommendation(detected, severity):
    if severity == 0 or len(detected) == 0:
        return (
            "### ✅ Cây khỏe mạnh\n"
            "Không phát hiện dấu hiệu bệnh rõ ràng.",
            "Tiếp tục theo dõi định kỳ 2 tuần/lần và duy trì điều kiện chăm sóc ổn định."
        )

    sev_names = ["Khỏe mạnh", "Rất nhẹ", "Nhẹ", "Nặng", "Rất nặng"]

    status = (
        f"### ⚠️ Phát hiện {len(detected)} bệnh\n"
        f"**Mức độ:** {severity}/4 — **{sev_names[severity]}**\n\n"
    )

    for d in detected:
        status += f"- {treatment_db[d]['title']}\n"

    advice = ""
    for d in detected:
        info = treatment_db[d]
        advice += f"### {info['title']}\n"
        advice += "**Biện pháp kỹ thuật**\n"
        for line in info["tech"].replace("•", "").split("\n"):
            line = line.strip()
            if line:
                advice += f"- {line}\n"

        advice += "\n**Khuyến nghị xử lý**\n"
        for line in info["chemical"].replace("•", "").replace("**", "").split("\n"):
            line = line.strip()
            if line:
                advice += f"- {line}\n"

        advice += "\n---\n"

    return status, advice

def predict(img, thr_miner, thr_rust, thr_phoma, thr_cerc):
    if img is None:
        return "📷 **Vui lòng tải ảnh lên**", ""

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    img_t = transform(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(img_t)

    # Use sigmoid for probability outputs
    d_probs = torch.sigmoid(out["disease_logits"])[0].cpu().numpy()
    s_probs = torch.sigmoid(out["severity_logits"])[0].cpu().numpy()

    p_healthy = d_probs[0]

    disease_probs = {
        "miner": d_probs[1],
        "rust": d_probs[2],
        "phoma": d_probs[3],
        "cercospora": d_probs[4]
    }

    thrs = {
        "miner": thr_miner,
        "rust": thr_rust,
        "phoma": thr_phoma,
        "cercospora": thr_cerc
    }

    detected = [
        d for d in DISPLAY_DISEASES
        if disease_probs[d] >= thrs[d]
    ]

    sev_level = int((s_probs > 0.5).sum())

    # Cập nhật logic chẩn đoán
    if p_healthy > 0.85 and len(detected) == 0:
        sev_level = 0

    if len(detected) > 0 and sev_level == 0:
        sev_level = 1

    status, advice = get_recommendation(detected, sev_level)
    return status, advice

custom_css = """
.gradio-container { background: #f0f4f0; }
.main-box { max-width: 1300px; margin: auto; }
.header { text-align: center; background: linear-gradient(135deg, #1B5E20, #4CAF50); padding: 20px; border-radius: 15px; margin-bottom: 20px; }
.header h1 { color: white; margin: 0; font-size: 28px; }
.header p { color: #c8e6c9; margin: 5px 0 0 0; }
.sidebar { background: white; border-radius: 15px; padding: 20px; height: 100%; }
.result-area { background: white; border-radius: 15px; padding: 20px; min-height: 500px; }
.status-box { background: #e8f5e9; padding: 15px; border-radius: 12px; border-left: 5px solid #4CAF50; margin-bottom: 20px; }
.advice-box { background: #fff8e1; padding: 15px; border-radius: 12px; border-left: 5px solid #FBC02D; }
.btn-analyze { background: linear-gradient(135deg, #2E7D32, #4CAF50) !important; border: none !important; color: white !important; font-weight: bold !important; font-size: 16px !important; height: 50px !important; margin-top: 15px !important; width: 100% !important; }
"""

with gr.Blocks(title="🌿 Coffee Clinic Pro", css=custom_css) as demo:
    with gr.Column(elem_classes="main-box"):
        gr.HTML("""
        <div class="header">
            <h1>🌿 COFFEE CLINIC V13</h1>
            <p>Hệ thống chẩn đoán bệnh lá cà phê đa nhiệm ứng dụng Deep Learning & CLIP</p>
        </div>
        """)

        with gr.Row():
            # Cột trái: Cấu hình
            with gr.Column(scale=2, elem_classes="sidebar"):
                gr.Markdown("## ⚙️ CẤU HÌNH AI")
                gr.Markdown("*Điều chỉnh ngưỡng nhạy cho từng bệnh*")

                s_miner = gr.Slider(0.1, 0.95, value=float(base_thrs[0]) if len(base_thrs) > 0 else 0.45,
                                    label="🐛 Sâu vẽ bùa (Miner)", step=0.01)
                s_rust = gr.Slider(0.1, 0.95, value=float(base_thrs[1]) if len(base_thrs) > 1 else 0.50,
                                   label="🍂 Rỉ sắt (Rust)", step=0.01)
                s_phoma = gr.Slider(0.1, 0.95, value=float(base_thrs[2]) if len(base_thrs) > 2 else 0.40,
                                    label="🌑 Bệnh Phoma", step=0.01)
                s_cerc = gr.Slider(0.1, 0.95, value=float(base_thrs[3]) if len(base_thrs) > 3 else 0.35,
                                   label="👁️ Đốm mắt cua (Cercospora)", step=0.01)

                gr.Markdown("---")
                gr.Markdown("**📌 Thông tin:**\n- Backbone: CLIP ViT-B/32\n- Task: Multi-label + Ordinal")

            # Cột giữa: Upload ảnh
            with gr.Column(scale=4):
                gr.Markdown("### 📸 ẢNH LÁ CÀ PHÊ")
                img_input = gr.Image(type="pil", label="", height=320)
                run_btn = gr.Button("🔬 CHẨN ĐOÁN NGAY", elem_classes="btn-analyze")

            # Cột phải: Kết quả
            with gr.Column(scale=4, elem_classes="result-area"):
                gr.Markdown("### 📊 KẾT QUẢ CHẨN ĐOÁN")

                with gr.Column(elem_classes="status-box"):
                    status_out = gr.Markdown("⏳ *Chờ dữ liệu đầu vào...*")

                with gr.Column(elem_classes="advice-box"):
                    gr.Markdown("#### 💡 PHÁC ĐỒ ĐIỀU TRỊ")
                    advice_out = gr.Markdown("*Kết quả sẽ hiển thị tại đây.*")

    run_btn.click(fn=predict, inputs=[img_input, s_miner, s_rust, s_phoma, s_cerc],
                  outputs=[status_out, advice_out])

if __name__ == "__main__":
    # Lưu ý: Chạy locally với share=False để ổn định, bật True nếu muốn tạo link công khai
    demo.launch(share=False)
