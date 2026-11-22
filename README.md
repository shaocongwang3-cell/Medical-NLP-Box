# 医疗 NLP 工具箱（Windows）

本项目提供医疗文本 NER、术语标准化、缩写扩展、拼写纠正和病历生成。已按 **Windows + Miniconda** 调整：向量库使用 **Chroma**（不再使用 Windows 不支持的 Milvus Lite）。

## 环境

- 已创建 conda 环境：`medical-nlp`（Python 3.11）
- 机器：NVIDIA GPU（CUDA 12.x）时安装 CUDA 版 PyTorch
- 前端：Node.js（建议 20 LTS；当前 Node 24 也可尝试）

激活环境：

```bat
call D:\miniconda3\condabin\conda.bat activate medical-nlp
```

## 安装依赖

```bat
call D:\miniconda3\condabin\conda.bat activate medical-nlp
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements_win.txt
cd frontend
npm install
```

## 构建术语向量库

首次使用「术语标准化」前，在项目根目录执行：

```bat
call D:\miniconda3\condabin\conda.bat activate medical-nlp
cd backend
python tools\create_chroma_db.py
```

会下载 `BAAI/bge-m3` 并把 `data/SNOMED_5000.csv` 写入 `backend/db/snomed_bge_m3`。

## 启动

后端（工作目录必须是 `backend`）：

```bat
scripts\start-backend.bat
```

或：

```bat
call D:\miniconda3\condabin\conda.bat activate medical-nlp
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

前端：

```bat
scripts\start-frontend.bat
```

浏览器打开 http://localhost:3000 ，后端 API 为 http://localhost:8000 。

## 功能说明

| 页面 | 是否依赖大模型 | 说明 |
|------|----------------|------|
| NER | 否 | HuggingFace `Clinical-AI-Apollo/Medical-NER` |
| 术语标准化 | 否 | Chroma + `BAAI/bge-m3` |
| 缩写 / 纠错 / 生成 | 是 | 当前未接 Ollama；接口会返回明确提示 |

大模型可稍后安装 Ollama，或配置 `OPENAI_API_KEY` 后在页面选择 OpenAI。
