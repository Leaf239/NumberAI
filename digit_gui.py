import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from model import NeuralNet

# 모델 불러오기
model = NeuralNet(784, [128], 10)

# 28x28 이미지를 확대해서 그리는 구조
img_size = 28
pixel_size = 10  # 28*10 = 280px
canvas_size = img_size * pixel_size

root = tk.Tk()
root.title("🧠 28x28 숫자 인식기")
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
canvas.pack()

# 이미지 버퍼: 실제 입력용 28x28
image = Image.new("L", (img_size, img_size), color=255)
draw = ImageDraw.Draw(image)

# 마우스로 그리기
def draw_callback(event):
    x, y = event.x // pixel_size, event.y // pixel_size
    r = 1  # 반지름 1 → 3x3 픽셀
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            if 0 <= x+dx < img_size and 0 <= y+dy < img_size:
                canvas.create_rectangle(
                    (x+dx)*pixel_size, (y+dy)*pixel_size,
                    (x+dx+1)*pixel_size, (y+dy+1)*pixel_size,
                    fill="black", outline=""
                )
                draw.point((x+dx, y+dy), fill=0)

canvas.bind("<B1-Motion>", draw_callback)

# 예측
def predict():
    arr = np.array(image).reshape(1, 784) / 255.0
    pred = model.forward(arr)
    digit = np.argmax(pred)
    result_label.config(text=f"예측 결과: {digit}")

# 지우기
def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, img_size, img_size], fill=255)
    result_label.config(text="")

# 버튼 UI
btn_frame = tk.Frame(root)
btn_frame.pack()
tk.Button(btn_frame, text="예측", command=predict).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="지우기", command=clear).pack(side=tk.LEFT, padx=5)
result_label = tk.Label(root, text="", font=("Arial", 24))
result_label.pack()

root.mainloop()