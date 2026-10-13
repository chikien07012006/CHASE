# Dữ liệu cho dự án CHASE

Tất cả lệnh chạy từ thư mục gốc của repo. `data/` nằm trong `.gitignore`, nên file này là nơi ghi lại cách lấy dữ liệu.

## Trạng thái (kiểm tra 12/10/2026)

| Mục | Trạng thái | Dùng cho việc gì trong dự án |
|---|---|---|
| `data/raw/primekg/kg_giant.csv` (852 MB) | ✅ có | Nguồn để dựng PrimeKG đầy đủ: graph cho PrimeKGQA và BioHopR, tính trần `ρ_H`, bảng khoảng cách theo kiểu `d_c`, ánh xạ tên sang node |
| `data/raw/rog_webqsp` (494 MB) | ✅ đủ: train 2,826 / val 246 / test 1,628 | So với UaG/CPR (Bảng 1); chạy thử CPR (cổng G1–G2); dữ liệu đầu tiên để thử DP và hiệu chỉnh |
| `data/raw/rog_cwq` (3.3 GB) | ✅ đủ 24 shard: train 27,639 / val 3,519 / test 3,531 (khớp với số CPR dùng) | So với UaG/CPR trên câu hỏi 2–4 hop (Bảng 1, cổng G4) |
| `models/all-MiniLM-L6-v2` (940 MB) | ✅ load được, dim 384 | Encoder cho câu hỏi, tên node, tên quan hệ (giống UaG/CPR). Chỉ cần `model.safetensors` + config; các thư mục `onnx/`, `openvino/` và file `tf_model.h5`, `rust_model.ot`, `pytorch_model.bin` (~850 MB) xoá được |
| `data/processed/primekg_graph` (39 MB) | ⚠️ graph bị cắt cạnh | Chỉ để tham khảo, **không** dùng cho thực nghiệm |
| BioHopR | ❌ chưa tải (13 MB) | Câu hỏi nhiều đáp án, kiểm chứng CRC (Định lý 2) |
| PrimeKGQA | ❌ chưa tải (1.55 GB) | Dataset y sinh chính: coverage, path vàng từ SPARQL, Mondrian/dịch chuyển phân phối |

## 0. Chuẩn bị (một lần)

```bash
source .venv/bin/activate
```

---

## Dataset

### 1. PrimeKG đầy đủ — ĐÃ CÓ
- **File:** `data/raw/primekg/kg_giant.csv` (8,196,862 dòng, 10 loại node, 30 quan hệ)
- **Mục đích:** dựng lại graph đầy đủ cho mọi thí nghiệm y sinh (PrimeKGQA, BioHopR). File có sẵn tên, kiểu, ID của node nên dùng luôn để ánh xạ thực thể.
- **Lưu ý:** **không** dùng `data/processed/primekg_graph/`. Graph đó chỉ có 6 loại node và 18 quan hệ, đồng thời `drug_drug` và `anatomy_protein_present` bị cắt còn 500k cạnh, nên sẽ làm mất đáp án.
- **Nếu cần tải lại:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM (file `kg.csv`)

### 2. WebQSP (bản RoG) — CẦN TẢI
- **Mục đích:** benchmark chính để so trực tiếp với UaG và CPR (Bảng 1 của bài). File có sẵn câu hỏi, thực thể chủ đề (`q_entity`), đáp án (`a_entity`) và subgraph Freebase cho từng câu (`graph`).
- **Dung lượng:** ~0.5 GB
- **Lệnh:**
  ```bash
  hf download rmanluo/RoG-webqsp --repo-type dataset --local-dir data/raw/rog_webqsp
  ```
- **Link:** https://huggingface.co/datasets/rmanluo/RoG-webqsp
- **Kiểm tra sau khi tải:** train khoảng 2,826–2,830 câu, test khoảng 1,628–1,630 câu.

### 3. CWQ (bản RoG) — CẦN TẢI
- **Mục đích:** như WebQSP, nhưng câu hỏi khó hơn (2–4 hop). So với UaG và CPR.
- **Dung lượng:** ~3.5 GB
- **Lệnh:**
  ```bash
  hf download rmanluo/RoG-cwq --repo-type dataset --local-dir data/raw/rog_cwq
  ```
- **Link:** https://huggingface.co/datasets/rmanluo/RoG-cwq
- **Kiểm tra sau khi tải:** trang HF ghi train 16,900 câu, nhưng CPR và RoG báo 27,639. Nếu lệch, phải lấy đúng split mà CPR dùng.

### 4. PrimeKGQA — CẦN TẢI (thủ công)
- **Mục đích:** dataset y sinh chính để đo coverage. Có SPARQL nên dựng được path vàng thật để train scorer, và có cấu trúc 2–4 node để chia nhóm theo số hop (Mondrian, dịch chuyển phân phối).
- **Dung lượng:** ~1.5 GB (3 file: `train_call_bioLLM.json`, `val_call_bioLLM.json`, `test_call_bioLLM.json`)
- **Link (tải thủ công):** https://zenodo.org/records/13829395 → lưu vào `data/raw/primekgqa/`
- **Hoặc bằng lệnh** (nếu link file trên Zenodo không đổi):
  ```bash
  mkdir -p data/raw/primekgqa && cd data/raw/primekgqa
  for f in train val test; do curl -L -O "https://zenodo.org/records/13829395/files/${f}_call_bioLLM.json"; done
  cd -
  ```
- **Giấy phép:** CC-BY-4.0
- **Kiểm tra sau khi tải:** tổng 83,999 câu. IRI trong SPARQL có thể không khớp ID trong `kg_giant.csv`; nếu vậy sẽ ánh xạ theo (tên, kiểu).
- **Code sinh dữ liệu (tham khảo):** https://github.com/xixi019/primeKGQG

### 5. BioHopR — CẦN TẢI
- **Mục đích:** câu hỏi nhiều đáp án (1–890 đáp án mỗi câu, trung bình 36.65), dùng để kiểm chứng kiểm soát tỉ lệ bỏ sót bằng CRC (Định lý 2).
- **Dung lượng:** ~13 MB
- **Lệnh:**
  ```bash
  hf download knowlab-research/BioHopR --repo-type dataset --local-dir data/raw/biohopr
  ```
- **Link:** https://huggingface.co/datasets/knowlab-research/BioHopR
- **Giấy phép:** MIT
- **Kiểm tra sau khi tải:** khoảng 7,633 dòng. Dataset chỉ có **tên thực thể**, không có ID, nên cần ánh xạ tên sang node PrimeKG (mục tiêu ≥95% khớp).

---

## Model

### 6. all-MiniLM-L6-v2 — CẦN TẢI
- **Mục đích:** encoder chính cho câu hỏi, tên node và tên quan hệ. Giống model UaG/CPR dùng, để so sánh công bằng.
- **Dung lượng:** ~90 MB
- **Lệnh:**
  ```bash
  hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2
  ```
- **Link:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

### 7. BGE-base — CẦN TẢI (cho ablation)
- **Mục đích:** ablation A8 (đổi encoder).
- **Dung lượng:** ~440 MB
- **Lệnh:**
  ```bash
  hf download BAAI/bge-base-en-v1.5 --local-dir models/bge-base-en-v1.5
  ```
- **Link:** https://huggingface.co/BAAI/bge-base-en-v1.5

### 8. Qwen3-8B — CẦN TẢI, NHƯNG Ở MÁY CÓ GPU
- **Mục đích:** chỉ baseline CPR cần, để sinh gợi ý quan hệ (1 lần gọi mỗi câu hỏi). CHASE không dùng LLM.
- **Dung lượng:** ~16 GB (bản AWQ khoảng 6 GB)
- **Yêu cầu:** CPR chạy model qua vLLM, mà vLLM cần GPU NVIDIA ≥24 GB (≥12 GB nếu dùng AWQ). **Không chạy được trên Mac.**
- **Lệnh (trên server GPU):**
  ```bash
  hf download Qwen/Qwen3-8B --local-dir models/Qwen/Qwen3-8B
  ```
- **Link:** https://huggingface.co/Qwen/Qwen3-8B

---

## Code baseline

### 9. CPR — CẦN CLONE
- **Mục đích:** baseline chính (nhóm N1). Đồng thời dùng để đo recall của TreeG (cổng G2).
- **Lệnh:**
  ```bash
  git clone https://github.com/betterCallSherlock/CPR third_party/CPR
  ```
- **Giấy phép:** MIT
- **Kiểm tra:** xem repo có ghi seed chia dữ liệu và file split không, để dùng đúng split CPR đã dùng.

### 10. SubgraphRAG — CẦN CLONE
- **Mục đích:** baseline "SubgraphRAG + conformal top-K" (nhóm N1). Code xử lý dữ liệu RoG cũng dùng lại được.
- **Lệnh:**
  ```bash
  git clone https://github.com/Graph-COM/SubgraphRAG third_party/SubgraphRAG
  ```
- **Giấy phép:** MIT

### 11. UaG — KHÔNG CÓ CODE
- **Mục đích:** baseline so sánh.
- **Tình trạng:** link https://github.com/Arstanley/UAG trả về 404 (kiểm tra 10/2026). Sẽ lấy số liệu UaG do CPR báo cáo, và tái hiện thủ tục của UaG trên scorer của CHASE (nhóm N2). Không cần tải gì.

---

## Không cần tải
- STaRK-Prime (đã bỏ khỏi phần chính)
- DDXPlus (hướng cũ)
- `kg.csv` gốc của PrimeKG (`kg_giant.csv` đã đủ)

## Tóm tắt dung lượng
| Mục | Dung lượng | Máy |
|---|---|---|
| Dataset 2–5 | ~5.5 GB | Mac |
| Model 6–7 | ~0.5 GB | Mac |
| Qwen3-8B | ~16 GB | Server GPU |
