# Baseline dùng chung scorer (`chase/baselines/`)

## Mục đích

Số liệu CPR và UaG trong README được lấy từ bài báo của họ. Hai paper này dùng scorer, retrieval và split khác với CHASE. Vì vậy khi CHASE thắng, ta chưa biết thắng nhờ **phương pháp** hay nhờ **scorer**.

Thư mục này trả lời câu hỏi đó. Mọi phương pháp ở đây dùng **cùng một scorer CHASE đã train**, cùng tập cal/test và cùng metric (ECR, APSS, tỉ lệ tập rỗng, re-split). Chúng chỉ khác nhau ở **cách tạo tập đáp án**. Không cần train lại.

## Các phương pháp

| Tên trong bảng | File | Cách tạo tập | Câu hỏi nó trả lời |
|---|---|---|---|
| **CHASE** | (dùng `chase.search.dp`) | Một ngưỡng τ̂ trên điểm gap g_v = σ_v − σ_min(q) (σ = bước tệ nhất của walk tốt nhất), tức trả về {v : σ_v ≤ σ_min(q) + τ̂}, tỉa ngay trong lúc search | Mốc so sánh |
| **Top-K** | `topk.py` | Xếp node theo σ, trả về K̂ node đầu. K̂ được calibrate bằng split conformal trên hạng của đáp án đúng tốt nhất | Ngưỡng thích ứng (tập to nhỏ tùy câu) có tốt hơn tập cố định K phần tử không? |
| **CPR-style B=b** | `beam.py` | Beam TreeG cố định như CPR: mỗi path mở rộng b bước tốt nhất, giữ b path tốt nhất mỗi hop. Sau đó lọc hậu kiểm bằng một ngưỡng conformal. Calibrate theo **cùng luật split conformal với CHASE** | Tỉa trong search (DP chính xác) có tốt hơn beam rồi mới lọc không? Beam làm trần coverage giảm bao nhiêu? |
| **CPR-style B=b, CPR rule** | `beam.py` | Cùng beam, nhưng calibrate theo **đúng luật trong code CPR** (`cpr/conformal/path_cp.py`), xem bên dưới | Tái hiện hành vi của CPR, đặc biệt là việc bao phủ dư |
| **UaG-style per-hop** | `per_hop.py` | Mỗi hop một ngưỡng riêng, cộng thêm một ngưỡng cho stop. Mỗi ngưỡng được calibrate ở mức α | Calibrate theo từng hop, không hiệu chỉnh, có bị trượt coverage không? |
| **UaG-style Bonferroni** | `per_hop.py` | Như trên, nhưng mỗi thành phần dùng mức α/(H+1) | Sửa cho hợp lệ (union bound) thì tập to lên bao nhiêu? |

### Chi tiết CPR-style (đọc từ repo `betterCallSherlock/CPR`)

- **Retrieval** theo `retrieve_candidates` trong `cpr/core.py`:
  - Mỗi path đang active mở rộng `branch` bước có điểm cục bộ tốt nhất.
  - Giữ `active` path có giá trị tốt nhất trên toàn bộ nhánh.
  - Ở mỗi hop, lấy node cuối của các path active làm ứng viên.
  - Mặc định của CPR là 32 × 32.
- **Ánh xạ sang scorer CHASE:**
  - Điểm cục bộ là điểm bước s_h.
  - Giá trị của path là bước tệ nhất.
  - Giá trị của một đáp án là max(giá trị path, λ·stop), giống DP.
  - Khi beam rộng vô hạn, kết quả trùng đúng với σ của DP (có test kiểm tra).
- **"CPR rule"**, đúng như code CPR:
  1. Điểm ∞ (đáp án đúng rơi khỏi beam) được thay bằng 1e9.
  2. τ̂ = `np.quantile` ở mức (n+1)(1−α)/n, có nội suy tuyến tính. Khi nhiều câu cal bị trượt, τ̂ gần 1e9, nghĩa là **trả cả beam** thay vì abstain.
  3. Nếu tập rỗng thì **thêm đáp án tốt nhất**.

  Luật 3 kéo ECR ở α cao lên gần Hit@1. Đây có thể là lý do CPR báo cáo ECR 50.8% khi mục tiêu chỉ là 20%. Bản chạy thử trên 60 câu cũng cho hiện tượng này.

### Chi tiết UaG-style (UaG không có code, nên đây là tái dựng)

- **Điểm calibrate:** với mỗi câu cal, lấy walk dẫn tới đáp án đúng tốt nhất (evidence walk của DP), rồi ghi các điểm thành phần [s₁, s₂, stop].
  - Hop nào walk không đi qua thì không tính.
  - Câu không có đáp án đi tới được thì ghi ∞ cho mọi thành phần.
- **Tính ngưỡng:** ngưỡng của mỗi thành phần là conformal quantile trên các câu có dùng thành phần đó.
- **Lúc test:** một node được trả về nếu có ít nhất một walk mà mọi bước s_h ≤ τ_h và stop ≤ τ_stop.
- **Phạm vi:** chỉ hỗ trợ H ≤ 2.
- **Ghi chú khi báo cáo:** cách UaG thật sự chia α cho các hop thì paper không cho đủ chi tiết. Vì vậy mình báo cáo cả hai biên: không hiệu chỉnh và Bonferroni.

## Đầu ra

`python -m chase.baselines.run` (chạy một model) ghi vào `--out`:
- `baselines.json`: mọi số liệu.
- `baselines.md`:
  - bảng cùng α trên split chính thức (ECR / APSS / % tập rỗng);
  - bảng cùng α theo trung bình re-split;
  - bảng cùng coverage (CHASE được calibrate tại ECR của từng baseline);
  - bảng chi phí (ms/câu, số cạnh được chấm điểm, tỉ lệ câu mà đáp án đúng còn trong tầm với, tức trần coverage của phương pháp).
- `fig_baselines_equal_coverage.png/pdf`: APSS theo coverage cho mọi phương pháp.

`python -m chase.baselines.report` (gộp nhiều seed) ghi `baselines_compare.md` (mean ± std) và hình tương ứng.

## Lưu ý khi đọc kết quả

- **Số lần re-split:** 1000 cho CHASE, Top-K và CPR-style; 200 cho UaG-style, vì mỗi lần re-split phải duyệt lại bảng điều kiện của từng câu.
- **Cùng coverage:** dùng ECR trung bình re-split của baseline làm mục tiêu cho CHASE. Hình chỉ vẽ các mức không có lần re-split nào bị abstain, ở mọi seed.
- **UaG trong hình:** bản per-hop được vẽ nét đứt với nhãn "no guarantee", vì nó không kiểm soát coverage và kích thước tập của nó chỉ là cận dưới. Bản Bonferroni trên CWQ abstain ở mọi mức trong ít nhất một seed, nên không vẽ được và hình có dòng chú thích cho việc này.
- **CHASE và Top-K không cho tập rỗng**, trừ khi câu hỏi không có ứng viên nào đi tới được (0% câu test WebQSP, 0.06% CWQ). Vì vậy coverage tối thiểu của chúng bằng Hit@1. Ở α cao, cả hai sẽ bao phủ dư. Trong bảng cùng coverage, dòng có '(floor)' là baseline có coverage thấp hơn mức sàn của CHASE, nên CHASE phủ nhiều hơn và tỉ lệ APSS không còn so ở cùng coverage.
- **Chi phí của CHASE** trong bảng là của DP không tỉa. Chi phí khi tỉa thật nằm trong `chase.eval.freebase`.

## Test

`tests/test_baselines.py` (84 test) kiểm tra các điểm sau:
- beam vô hạn cho kết quả bằng DP, còn beam hẹp chỉ làm mất path;
- hạng của Top-K tính đúng;
- bảng điều kiện per-hop khớp với kết quả liệt kê mọi walk;
- các thành phần của walk dựng lại đúng σ*;
- Bonferroni đạt coverage còn per-hop chỉ được khoảng (1−α)³ trong mô phỏng;
- luật CPR khớp với code CPR.
