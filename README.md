# U2Net Background Remover (FastAPI)

A FastAPI-based API for background removal using U2Net.  
Supports multiple concurrent requests and returns PNG images with transparency.

## 🚀 Features
- Uses **U2Net ONNX** model
- Keeps **all objects** (person, clothes, cars, pets, etc.)
- **Refined masks** for smooth edges
- Async FastAPI server with multiple workers support

## 📦 Installation
Clone repo:
```bash
git clone https://github.com/<your-username>/background-remover.git
cd background-remover
