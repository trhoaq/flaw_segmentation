# railFlaw-segmentation

Chuẩn hóa mặt nạ lỗi bề mặt ray và huấn luyện **YOLO26 instance segmentation**, mặc định `yolo26n-seg.pt`.

Code: `D:\Code\railFlaw-segmentation`. Dữ liệu chuyển đổi: `D:\dataset\rail\yolo`.
Code đọc ảnh/mặt nạ gốc và ghi bản sao cùng nhãn TXT polygon vào `yolo`; không sửa ảnh, nhãn hay ZIP gốc.

## Chạy trên Windows

```powershell
cd D:\Code\railFlaw-segmentation
.\setup.ps1
.\.venv\Scripts\python.exe prepare.py
.\.venv\Scripts\python.exe validate.py
.\.venv\Scripts\python.exe train.py --smoke
.\.venv\Scripts\python.exe train.py
```

Không cần activate môi trường. `setup.ps1` tạo `.venv` và cài thư viện; `requirements-lock.txt` ghi môi trường đã kiểm thử nếu có. Để cài lại chính xác: `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`.

## Nguồn và chuẩn hóa

| Nguồn | Mặt nạ | Cặp ảnh–mask dự kiến |
|---|---|---:|
| RSDDs Type-I và Type-II | JPG đen/trắng, threshold >127 | 67 + 128 |
| NRSD-relabel | Chỉ số palette 0=nền, 1=lỗi | 4.101 |
| NRSD bản nhỏ | Như trên; chuyển đổi riêng để đối chiếu | 4.101 |
| NEU RSDDS-113 | Nhị phân, đi kèm ảnh màu và độ sâu | 113 |
| CGAN | Đỏ=lỗi, xanh=nền; dữ liệu tổng hợp | 3.000 |
| Rail-5k sample | 10 ảnh có mask `_LW` raster hóa từ bbox; chỉ để tham khảo | 10 |

Mọi vùng foreground ánh xạ về lớp **0: rail_surface_defect**. Bộ dữ liệu gốc không cung cấp một hệ nhãn đa lớp thống nhất, nên code không suy đoán loại lỗi từ tên ảnh. Các thành phần liên thông của semantic mask được dùng làm instance dẫn xuất; chúng không phải instance ID do chuyên gia gán.

Các bộ không có segmentation được tách nguyên thư mục sang `D:\dataset\rail_excluded_no_segmentation` bằng `filter_originals.py --apply`. Lệnh không xóa hay sửa nội dung tệp; `RESTORE_INFO.json` lưu đường dẫn khôi phục. Rail-5k gốc vẫn có cả ảnh raw và nhãn thiếu ảnh; bộ chuyển đổi chỉ lấy 10 cặp hợp lệ. Metadata macOS trong CGAN không được coi là ảnh.

## Cấu trúc đầu ra

```text
D:/dataset/rail/yolo/
  data.yaml              # train real + synthetic train; val/test chỉ ảnh thật
  data_real.yaml         # chỉ ảnh thật
  train.txt, val.txt, test.txt
  manifest.csv           # nguồn, nhóm, split, hash, số polygon và IoU chuyển đổi
  conversion_summary.json
  validation_report.json
  RSDDs_Type_1/images/{train,val,test}/
  RSDDs_Type_1/labels/{train,val,test}/
  ... các bộ khác có cấu trúc tương tự và data.yaml riêng
```

NRSD bản nhỏ và relabel đều được chuyển đổi, nhưng cấu hình tổng hợp chỉ dùng relabel để tránh tính hai lần cùng bộ. Các crop NRSD cùng ảnh gốc được giữ chung split. CGAN và RSDDs nguồn có chung nhóm; phiên bản/ảnh trùng hash không vượt ranh giới split. Seed mặc định 42. CGAN thuộc val/test được giữ trong thư mục riêng nhưng không dùng để đánh giá cấu hình tổng hợp.

Mask Rail-5k mẫu khớp chính xác phép tô các bounding box `LW` trong XML; chúng không phải biên lỗi được vẽ chi tiết. `Rail5k_sample/` vẫn có bản chuyển đổi tham khảo, nhưng mặc định **không có trong data.yaml hoặc data_real.yaml**. `configs/prepare.yaml` đặt `include_coarse_rail5k: false`. Muốn thay đổi chỉ danh sách huấn luyện, không chạy lại chuyển đổi: `python refresh_catalogs.py --include-coarse-rail5k`; khôi phục mặc định bằng `python refresh_catalogs.py`.

Nhãn YOLO-seg: `0 x1 y1 x2 y2 ...`, tọa độ chuẩn hóa [0,1], tối thiểu 3 đỉnh. Code dùng contour trên lưới 2x để giữ vùng 1 pixel, nối các vòng contour để biểu diễn lỗ, rồi rasterize nhãn TXT ngược lại và kiểm tra IoU >=0,995 cho từng ảnh. Mask rỗng có TXT rỗng, là mẫu nền có nhãn hợp lệ. Ảnh đầu ra là bản sao nguyên byte.

## Huấn luyện và đánh giá

Sửa `configs/train.yaml` hoặc dùng tham số:

```powershell
.\.venv\Scripts\python.exe train.py --model yolo26s-seg.pt --epochs 100 --imgsz 640 --batch 8 --device 0
.\.venv\Scripts\python.exe train.py --data D:/dataset/rail/yolo/data_real.yaml
.\.venv\Scripts\python.exe train.py --resume runs/yolo26n_rail_seg/weights/last.pt
.\.venv\Scripts\python.exe evaluate.py --weights runs/yolo26n_rail_seg/weights/best.pt --split test
.\.venv\Scripts\python.exe predict.py --weights runs/yolo26n_rail_seg/weights/best.pt --source D:/path/to/images
```

`device: auto` dùng CUDA nếu PyTorch hiện tại hỗ trợ, nếu không dùng CPU. Nếu có GPU NVIDIA nhưng cài bản PyTorch CPU, cài lại PyTorch CUDA theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/). `workers: 0` tương thích Windows. Ảnh ray dài và mảnh có thể cần `imgsz` lớn hơn để giữ lỗi nhỏ.

`--smoke` chỉ chạy 1 epoch CPU trên 8 ảnh train và 4 ảnh val để kiểm tra kỹ thuật; checkpoint này chưa phải mô hình được huấn luyện đầy đủ. Kết quả thật nằm ở `runs/`, báo cáo môi trường và smoke test ở `reports/`. Chưa có cơ sở so sánh độ chính xác nếu chưa train và đánh giá trên tập test được giữ riêng.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe doctor.py
.\.venv\Scripts\python.exe preview.py
```

Tài liệu: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [YOLO segmentation format](https://docs.ultralytics.com/datasets/segment/), [training](https://docs.ultralytics.com/modes/train/). Nguồn và giấy phép dữ liệu vẫn theo `SOURCE.json`/`SOURCE.txt` của từng bộ gốc.
