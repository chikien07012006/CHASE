# Chạy CHASE trên BioHopR

BioHopR (Hugging Face `knowlab-research/BioHopR`) là bộ câu hỏi y sinh nhiều hop trên PrimeKG. File này mô tả:
- cách đưa bộ dữ liệu vào pipeline của CHASE;
- bước nào khác so với WebQSP/CWQ;
- mỗi bước dùng file nào và ghi ra file nào.

**Tóm tắt:** chỉ có **bước tạo store** là khác. Store của BioHopR được ghi theo đúng định dạng store của Freebase, nên các bước text cache, splits, gold path, train, eval và baseline cùng scorer chạy lại y nguyên lệnh cũ.

## 1. Đầu vào

| File | Nội dung | Có từ đâu |
|---|---|---|
| `data/raw/biohopr/BioHopR.json` | 7,633 dòng. Mỗi dòng có `hop2` (topic của câu 2 hop), `hop1` (bridge, cũng là topic của câu 1 hop), `answer` (danh sách **tên**), kiểu của từng thực thể, và hai câu hỏi `hop1_question` / `hop2_question` | Tải từ Hugging Face |
| `data/processed/primekg_full/` (`csr.npz`, `nodes.parquet`, `meta.json`) | PrimeKG đầy đủ: 134,437 node, 10 kiểu, 31 quan hệ, 8.2 triệu cạnh có hướng (mỗi cạnh vô hướng lưu cả hai chiều) | `chase.kg.build_primekg`, đã build |
| `data/processed/primekg_full/type_dist.npz` | Khoảng cách từ mỗi node tới kiểu gần nhất, dùng để lọc theo kiểu | `chase.kg.type_distance`, đã build |

## 2. Khác gì so với WebQSP / CWQ

| Bước | WebQSP / CWQ | BioHopR |
|---|---|---|
| **Tạo store** | `chase.kg.build_freebase`: RoG có sẵn subgraph 2 hop cho từng câu, đáp án đã là ID node | **`chase.data.biohopr`**: khớp tên sang node, tự cắt subgraph từ PrimeKG, tự chia tập (mục 3) |
| Text cache | `chase.model.text_cache freebase` | Giống |
| Splits | WebQSP lấy cal từ train. CWQ dùng `--cal-from validation` | `--cal-from validation`, giống CWQ |
| ρ_H, gold path | `chase.search.dp`, `chase.data.gold_paths` | Giống |
| Train | WebQSP α = 0.1. CWQ α = 0.3 (ρ ≈ 0.80) | α = 0.1 (ρ₂ ≈ 0.955) |
| Eval | Có số CPR/UaG reported để so | **Không có số reported**, nên bảng compare chỉ có CHASE |
| Baseline cùng scorer | `chase.baselines.run`, `report` | Giống. Đây là **phép so sánh chính** trên BioHopR |

## 3. Bước tạo store: `chase/data/biohopr.py`

Lệnh: `python -m chase.data.biohopr --out data/processed/biohopr`, chạy khoảng 1.5 phút.

### 3.1 Câu hỏi (`load_questions`)
- **Câu 2 hop:** mỗi dòng cho một câu. Topic là `hop2`, kiểu đáp án là `target_type`.
- **Câu 1 hop:** topic là `hop1`. Nhiều dòng có chung câu 1 hop, nên các câu này được gộp theo nội dung câu hỏi và hợp các tập đáp án.
- Tổng cộng 7,633 câu 2 hop và 2,494 câu 1 hop, tức **10,127 câu**.
- Dùng cách hỏi số ít ("Name a disease …"). Câu này hỏi **ít nhất một** đáp án, khớp với mục tiêu coverage (ECR) của CHASE.

### 3.2 Khớp tên sang node (`NameIndex`)
- Chỉ khớp với node **đúng kiểu** đã ghi trong dữ liệu.
- Thứ tự thử: khớp chính xác tên chính, nếu không được thì so chữ thường với tên chính hoặc alias.
- Kết quả:
  - topic: 10,050 khớp chính xác, 2 khớp nhờ chữ thường/alias, **75 không khớp**;
  - đáp án: 343,855 khớp chính xác, 54 khớp nhờ chữ thường/alias, **5,729 không khớp** (khoảng 1.6%).
- Câu không khớp được topic vẫn được giữ lại. Câu đó không thể có đáp án đi tới được, nên luôn tính là trượt, vì bảo đảm coverage áp dụng cho mọi câu. Đáp án không khớp được tính là "absent".

### 3.3 Subgraph có lọc kiểu (`typed_subgraph`)
Trên toàn PrimeKG, nhiều topic có hơn 1 triệu cạnh trong 2 hop, quá nặng để train. Mỗi câu chỉ giữ:
- **hop 1:** cạnh từ topic tới các node cách kiểu đáp án ≤ 1 bước (tra trong `type_dist.npz`);
- **hop 2:** cạnh từ các node hop 1 tới node **đúng kiểu đáp án**;
- cả hai chiều của mỗi cạnh, giống PrimeKG và store Freebase. Bước tìm gold path cần đi ngược từ đáp án.

Mọi walk ≤ 2 hop kết thúc ở một node đúng kiểu đáp án đều nằm trong subgraph này, nên **σ của mọi node đúng kiểu bằng đúng σ trên toàn đồ thị**. `tests/test_biohopr.py` kiểm tra điều này trên 30 đồ thị ngẫu nhiên.

Dùng kiểu đáp án là hợp lệ, vì câu hỏi tự nêu kiểu ("Name a *drug* …"). Đây chính là ràng buộc type feasibility trong proposal.

Kích thước (số cạnh có hướng):

| Phần | Trung vị | p90 | Lớn nhất |
|---|---|---|---|
| train | 576 | 24,316 | 1,522,850 |
| validation | 572 | 29,222 | 1,491,832 |
| test | 414 | 32,764 | 632,134 |

Phần lớn câu nhỏ hơn WebQSP (trung vị khoảng 8k cạnh). Một số ít câu có topic là hub, ví dụ "drug có side effect Headache", có subgraph rất lớn: 22 câu train trên 100k cạnh.

### 3.4 Kiểu node
- `node_type = 0`: node **đúng kiểu đáp án**, được phép làm đáp án.
- `node_type = 1`: mọi kiểu khác.

Quy ước này giống bên Freebase (0 = entity, 1 = CVT không được làm đáp án), nên DP, sampler, eval và baseline chạy mà không phải sửa. Topic không bao giờ là đáp án; DP đã xử lý việc này.

### 3.5 Chia tập (`split_by_group`)
- BioHopR không có split chính thức, nên chia **70 / 15 / 15** (train / validation / test) theo **topic**. Không topic nào nằm ở hai phần. Seed chia tập cố định là 0, độc lập với seed train.
- `qid = "G<nhóm>_<k>"`, nên `chase.data.splits` cũng giữ nguyên nhóm khi lấy 10% train làm tập early stopping.

| Phần | Số câu | Số topic | Có đáp án trong subgraph | Không khớp được topic |
|---|---|---|---|---|
| train | 7,089 | 6,036 | 96.4% | 52 |
| validation (dùng làm cal) | 1,520 | 1,291 | 96.5% | 10 |
| test | 1,518 | 1,269 | 95.5% | 13 |

### 3.6 Đầu ra (`data/processed/biohopr/`)

| File | Nội dung |
|---|---|
| `{train,validation,test}/csr.npz` | Subgraph của mọi câu, gộp thành mảng phẳng (`node_offset`, `edge_offset`, `indptr`, `indices`, `rel`, `node_type`) |
| `{train,validation,test}/questions.parquet` | Mỗi câu một dòng: `qid`, `question`, `q_entity`, `a_entity` (tên), `topic_nodes`, `answer_nodes` (ID cục bộ), `hops`, `answer_type`, `pattern` (chuỗi kiểu, ví dụ `drug:gene/protein:disease`), `topic_key` (khóa node PrimeKG) |
| `{train,validation,test}/node_names.parquet` | Tên node của từng subgraph, dùng cho text cache |
| `relations.json` | 31 quan hệ của PrimeKG. Không có quan hệ ngược riêng, vì PrimeKG vô hướng |
| `stats.json` | Số câu, kết quả khớp tên, kích thước subgraph theo từng phần |

## 4. Các bước sau: dùng lại nguyên code

| Bước | Module | Đọc | Ghi |
|---|---|---|---|
| Text cache | `chase.model.text_cache freebase` | `node_names.parquet`, `questions.parquet`, `relations.json` | `emb_minilm/` |
| Splits | `chase.data.splits --cal-from validation` | `questions.parquet` | `splits.json`: cal = validation (1,520), val = 10% train (early stopping), train = phần còn lại, test = test |
| Kiểm tra ρ_H | `chase.search.dp` | store | In ra ρ₁ ≈ 27.6%, ρ₂ ≈ 95.5% (test) |
| Gold path | `chase.data.gold_paths` | store, `emb_minilm/` | `gold_paths/train.parquet`. Chạy thử: 97% câu train có gold path |
| Train | `chase.train.train` | store, `splits.json`, `gold_paths/` | `runs/biohopr_seed*/best.pt`, `history.json` |
| Eval | `chase.eval.freebase --dataset biohopr` | `best.pt`, cal, test | `runs/biohopr_seed*/eval/`: `metrics.json`, `validity.json`, hình |
| Gộp seed | `chase.eval.compare` | các thư mục `eval/` | `runs/biohopr_compare/compare.md` (chỉ có CHASE) |
| Baseline cùng scorer | `chase.baselines.run`, `chase.baselines.report` | `best.pt`, cal, test | `runs/biohopr_seed*/baselines/`, `runs/biohopr_baselines/` |

## 5. Lưu ý

- **Mục tiêu coverage** vẫn là "ít nhất một đáp án", như Freebase. Mỗi câu có trung bình 36.6 đáp án (trung vị 11), nên mục tiêu này có thể dễ. Xem kết quả rồi quyết định có thêm CRC (tỉ lệ bỏ sót) hay không.
- **Thiếu nhãn:** đáp án chỉ đi theo **một** bridge. Câu 2 hop không nhắc tên bridge, nên các node đúng kiểu đi qua bridge khác bị tính là sai. APSS vì vậy có thể bị phạt oan.
- **Câu có subgraph rất lớn** (topic là hub) làm chậm stage 2 và eval, nhưng số lượng ít.
- **Không so được với số reported:** CPR/UaG không chạy BioHopR. Phép so sánh là các baseline cùng scorer.

## 6. Lệnh chạy

```bash
source .venv/bin/activate
caffeinate -dims -w $$ &          # giữ máy và màn hình thức tới khi đóng terminal này
D=data/processed/biohopr

# 1. Dữ liệu
python -m chase.data.biohopr --out $D
python -m chase.model.text_cache freebase $D
python -m chase.data.splits --data $D --cal-from validation
python -m chase.search.dp --data $D --split test --H 1 2
python -m chase.data.gold_paths --data $D --split train --H 2

# 2. Train 3 seed
for s in 0 1 2; do
  python -m chase.train.train --data $D --out runs/biohopr_seed$s --seed $s 2>&1 | tee runs/biohopr_seed$s.train.log
done

# 3. Eval + gộp seed
for s in 0 1 2; do
  python -m chase.eval.freebase --data $D --model runs/biohopr_seed$s/best.pt \
         --out runs/biohopr_seed$s/eval --dataset biohopr --main-alpha 0.1
done
python -m chase.eval.compare --evals runs/biohopr_seed{0,1,2}/eval --out runs/biohopr_compare

# 4. Baseline cùng scorer
for s in 0 1 2; do
  python -m chase.baselines.run --data $D --model runs/biohopr_seed$s/best.pt \
         --out runs/biohopr_seed$s/baselines --beams 8 32 128
done
python -m chase.baselines.report --runs runs/biohopr_seed{0,1,2}/baselines --out runs/biohopr_baselines
```
