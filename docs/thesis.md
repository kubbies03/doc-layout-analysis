# PHÂN TÍCH BỐ CỤC TÀI LIỆU PDF SỬ DỤNG HỌC SÂU
### *Document Layout Analysis using Deep Learning*

**Sinh viên thực hiện:** LÊ TIẾN THẮNG  
**Mã số sinh viên:** 22200144  
**Giảng viên hướng dẫn:** ThS. Nguyễn Quốc Khoa  
**Khoa:** Điện Tử - Viễn Thông — Bộ môn Máy Tính – Hệ Thống Nhúng  
**Trường:** Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM  
**Năm học:** 2025–2026

---

# LỜI CẢM ƠN

Em xin gửi lời cảm ơn chân thành đến Khoa Điện Tử - Viễn Thông, Bộ môn Máy Tính – Hệ Thống Nhúng, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM đã tổ chức chương trình đào tạo và tạo môi trường học thuật để em có nền tảng thực hiện luận văn này.

Đặc biệt, em xin phép được gửi lời cảm ơn sâu sắc đến **ThS. Nguyễn Quốc Khoa** đã tận tình hướng dẫn, góp ý và chỉnh sửa trong suốt quá trình nghiên cứu và viết luận văn. Sự định hướng và hỗ trợ của Thầy là nguồn động lực lớn giúp em hoàn thiện đề tài.


Do kiến thức và kinh nghiệm còn hạn chế, luận văn không tránh khỏi những thiếu sót. Em kính mong nhận được sự góp ý quý báu của Thầy/Cô và Hội đồng để em có thể hoàn thiện đề tài tốt hơn.

---

# TÓM TẮT

Tài liệu PDF không được thiết kế để máy tính đọc hiểu — nội dung bên trong là một bức tranh bố cục hai chiều mà máy chỉ thấy từng điểm ảnh, không biết đâu là tiêu đề, đâu là bảng số liệu, đâu là chú thích. Luận văn này giải quyết vấn đề đó bằng cách xây dựng một hệ thống tự động phân tích cấu trúc trang tài liệu từ đầu đến cuối, biến một trang PDF thành dữ liệu có tổ chức mà các ứng dụng khác có thể sử dụng trực tiếp.

Hệ thống hoạt động qua bốn giai đoạn nối tiếp. Đầu tiên, một mô hình học sâu quan sát toàn bộ trang và khoanh vùng từng thành phần — phân biệt được tiêu đề, đoạn văn, bảng biểu, hình ảnh, công thức và các loại vùng khác. Tiếp theo, văn bản trong từng vùng được nhận dạng bằng công nghệ OCR, trong khi bảng biểu và công thức — vốn quá phức tạp với OCR thông thường — được xử lý bởi một mô hình thị giác chuyên biệt. Sau đó, các vùng được sắp xếp theo thứ tự đọc tự nhiên của con người, không đơn giản là từ trên xuống dưới mà phải xử lý đúng bố cục nhiều cột. Cuối cùng, toàn bộ nội dung được đưa qua một mô hình ngôn ngữ lớn để tổ chức thành cấu trúc có ý nghĩa — gom nhóm các đoạn văn vào đúng mục, nhận diện quan hệ giữa bảng và phần văn bản liên quan, tạo tóm tắt và từ khóa.

Kết quả thực nghiệm cho thấy hệ thống đạt hiệu suất tốt trên tập dữ liệu DocLayNet — một bộ dữ liệu chuẩn gồm các tài liệu thực tế đa dạng từ báo cáo tài chính đến văn bản pháp luật. Khả năng phát hiện vùng bố cục đạt mức cạnh tranh với các phương pháp hiện đại, kể cả với những thành phần nhỏ và hiếm như chú thích hay tiêu đề. Chất lượng nhận dạng văn bản đủ tốt cho các ứng dụng tìm kiếm và phân tích nội dung. Đặc biệt, giai đoạn làm giàu ngữ nghĩa cho thấy mô hình ngôn ngữ lớn có thể tuân theo schema đầu ra một cách đáng tin cậy và tái tạo đúng nội dung bảng biểu — hai yếu tố then chốt khi triển khai thực tế. Toàn bộ pipeline có thể chạy trên phần cứng phổ thông với chi phí vận hành thấp, phù hợp cho việc xử lý tài liệu ở quy mô vừa và lớn.

**Từ khóa:** Trí tuệ nhân tạo, phân tích bố cục tài liệu, YOLO, OCR, mô hình ngôn ngữ lớn, DocLayNet.

---

# ABSTRACT

PDF documents are not designed for machine comprehension — their content is a two-dimensional layout canvas where text, tables, figures, and formulas are arranged according to visual logic that computers cannot directly interpret. This thesis addresses that problem by constructing an end-to-end pipeline that automatically analyzes the structure of document pages and transforms raw PDF pages into organized, machine-readable data.

The system operates through four sequential stages. First, a deep learning model scans the entire page and delineates each content region, distinguishing between titles, paragraphs, tables, figures, formulas, and other layout elements. Second, text within each region is recognized using an OCR engine, while tables and formulas — too complex for standard OCR — undergo specialized processing. Third, the detected regions are ordered according to natural human reading flow, handling multi-column layouts correctly rather than naively reading top-to-bottom. Finally, all content is passed through a large language model that organizes the material into a meaningful hierarchical structure — grouping paragraphs into sections, identifying relationships between tables and their surrounding text, and generating summaries and keywords.

Experimental results on DocLayNet — a benchmark dataset of real-world documents including financial reports, scientific papers, and legal texts — demonstrate competitive performance across all pipeline stages. Layout detection achieves mAP@0.5 = 0.909 on the test set across all 11 document element classes. The semantic enrichment stage reaches a schema parse rate of 99.8%, content recall of 0.800, and table token F1 of 0.870 on 485 samples, at a processing cost of approximately \$0.0038 per page. The complete pipeline runs on commodity hardware, making it suitable for medium-to-large scale document processing without requiring specialized infrastructure.

**Keywords:** Artificial Intelligence, document layout analysis, YOLO, OCR, large language model, DocLayNet.

---

# MỤC LỤC

- [Chương 1: Giới thiệu](#chương-1)
  - [1.4 Đóng góp chính](#)
  - [1.5 Câu hỏi nghiên cứu](#)
  - [1.6 Cấu trúc luận văn](#)
- [Chương 2: Cơ sở lý thuyết](#chương-2)
- [Chương 3: Phương pháp và triển khai](#chương-3)
- [Chương 4: Kết quả thực nghiệm](#chương-4)
- [Chương 5: Kết luận](#chương-5)
  - [5.1 Tóm tắt kết quả đạt được](#)
  - [5.2 Đóng góp của luận văn](#)
  - [5.3 Kỹ năng và kiến thức đạt được](#)
  - [5.4 Đối chiếu với mục tiêu ban đầu](#)
  - [5.5 Hạn chế](#)
  - [5.6 Hướng phát triển](#)
- [Tài liệu tham khảo](#tltk)
- [Phụ lục](#phu-luc)
  - [Phụ lục A: Cấu hình training đầy đủ](#)
  - [Phụ lục B: Ví dụ output JSON](#)
  - [Phụ lục C: Kết quả benchmark OCR đầy đủ](#)
  - [Phụ lục D: Đánh giá LLM offline — Phi-3 Mini 4-bit](#)
  - [Phụ lục E: Prompt LLM đầy đủ](#)

---

# DANH MỤC HÌNH

| Số | Tên hình | Trang |
|----|---------|-------|
| Hình 3.1 | Kiến trúc tổng thể pipeline bốn giai đoạn | Ch3 |
| Hình 3.2 | Phân phối 11 lớp đối tượng trong DocLayNet (tập huấn luyện) | Ch3 |
| Hình 3.3 | Phân phối tỷ lệ khung hình (AR) và kích thước bounding box | Ch3 |
| Hình 3.4 | Sơ đồ thiết kế ablation study — 3 thí nghiệm | Ch3 |
| Hình 3.6 | Sơ đồ định tuyến OCR theo nhãn khối | Ch3 |
| Hình 3.7 | Thuật toán XY-Cut tùy chỉnh — 4 bước | Ch3 |
| Hình 3.8 | Kết quả phát hiện bố cục YOLOv11s trên tài liệu thực tế | Ch3 |
| Hình 3.9 | Minh họa 4 giai đoạn pipeline trên trang PDF mẫu | Ch3 |
| Hình 3.10 | Grid các khối crop từ trang mẫu theo từng nhãn lớp | Ch3 |
| Hình 3.11 | Ví dụ kết quả OCR docTR trên khối văn bản | Ch3 |
| Hình 4.1 | Framework đánh giá 3 cấp độ | Ch4 |
| Hình 4.2 | Đường cong hội tụ training — 3 thí nghiệm ablation | Ch4 |
| Hình 4.3 | So sánh tổng thể 3 thí nghiệm ablation (val + test) | Ch4 |
| Hình 4.4 | AP@0.5 từng lớp trên tập kiểm định | Ch4 |
| Hình 4.5 | AP@0.5 từng lớp trên tập kiểm tra | Ch4 |
| Hình 4.6 | So sánh hiệu quả oversampling trên lớp hiếm | Ch4 |
| Hình 4.7 | F1 phát hiện khối theo từng lớp — đánh giá end-to-end | Ch4 |
| Hình 4.8 | CER nhận dạng văn bản theo từng lớp | Ch4 |
| Hình 4.9 | Confusion matrix — YOLOv11s trên tập kiểm tra | Ch4 |
| Hình 4.10 | Số vật thể mỗi trang trong DocLayNet | Ch4 |
| Hình 4.11 | Ví dụ kết quả dự đoán YOLO trên tập kiểm tra | Ch4 |
| Hình 4.12 | Trang gốc mẫu thử — minh họa end-to-end | Ch4 |
| Hình 4.13 | DocLayNet Ground Truth annotations mẫu thử | Ch4 |

---

# DANH MỤC BẢNG

| Số | Tên bảng | Trang |
|----|---------|-------|
| Bảng 2.1 | So sánh các tập dữ liệu Document Layout Analysis | Ch2 |
| Bảng 2.2 | So sánh các phương pháp SOTA trên DocLayNet | Ch2 |
| Bảng 3.1 | Phân phối annotation 11 lớp trong DocLayNet (tập train) | Ch3 |
| Bảng 3.2 | Thống kê kích thước và tỷ lệ khung hình bounding box | Ch3 |
| Bảng 3.3 | Cấu hình huấn luyện cố định giữa các thí nghiệm | Ch3 |
| Bảng 3.4 | Kết quả benchmark 3 engine OCR trên 50 mẫu DocLayNet | Ch3 |
| Bảng 3.5 | So sánh thuật toán reading order | Ch3 |
| Bảng 3.6 | So sánh đầy đủ cấu hình 3 thí nghiệm ablation | Ch3 |
| Bảng 4.1 | Kết quả so sánh 3 thí nghiệm ablation (tập kiểm định) | Ch4 |
| Bảng 4.2 | Kết quả so sánh 3 thí nghiệm ablation (tập kiểm tra) | Ch4 |
| Bảng 4.3 | AP@0.5 từng lớp — 3 thí nghiệm (kiểm định) | Ch4 |
| Bảng 4.4 | AP@0.5 từng lớp — 3 thí nghiệm (kiểm tra) | Ch4 |
| Bảng 4.5 | Kết quả phát hiện khối end-to-end trên 485 mẫu | Ch4 |
| Bảng 4.6 | Chất lượng OCR (CER) theo từng lớp | Ch4 |
| Bảng 4.7 | Kết quả giai đoạn LLM trên 485 mẫu | Ch4 |
| Bảng 4.7b | Phân tích error budget — nguồn gốc 20% content recall thất thoát | Ch4 |
| Bảng 4.8 | Tổng hợp pattern lỗi định tính theo lớp | Ch4 |
| Bảng 4.9 | Tổng hợp kết quả so với mục tiêu đề ra | Ch4 |
| Bảng 4.10 | Latency từng module pipeline (benchmark 50 mẫu) | Ch4 |
| Bảng 4.11 | Thống kê token và chi phí Gemini 2.5 Flash | Ch4 |
| Bảng 4.12 | So sánh chi phí xử lý tài liệu với các giải pháp thay thế | Ch4 |
| Bảng 4.13 | So sánh chất lượng pipeline với các phương pháp baseline (100 mẫu) | Ch4 |
| Bảng 4.14 | Tổng kết kết quả toàn pipeline — các chỉ số chính | Ch4 |

---

# DANH SÁCH CHỮ VIẾT TẮT

| Viết tắt | Giải thích |
|---------|-----------|
| ACL | Association for Computational Linguistics — Hiệp hội Ngôn ngữ học Tính toán |
| ACM | Association for Computing Machinery — Hiệp hội Máy tính |
| AI | Artificial Intelligence — Trí tuệ nhân tạo |
| AP | Average Precision — Độ chính xác trung bình |
| API | Application Programming Interface — Giao diện lập trình ứng dụng |
| AR | Aspect Ratio — Tỷ lệ khung hình |
| BEiT | BERT pre-training of Image Transformers — Mô hình Transformer ảnh huấn luyện trước |
| BERT | Bidirectional Encoder Representations from Transformers — Mô hình ngôn ngữ hai chiều |
| BiLSTM | Bidirectional Long Short-Term Memory — Mạng nhớ dài ngắn hạn hai chiều |
| BLEU | Bilingual Evaluation Understudy — Chỉ số đánh giá văn bản sinh |
| BnB | BitsAndBytes — thư viện lượng tử hóa 4-bit |
| CER | Character Error Rate — Tỷ lệ lỗi ký tự |
| CJK | Chinese–Japanese–Korean — Bộ ký tự CJK |
| CNN | Convolutional Neural Network — Mạng nơ-ron tích chập |
| COCO | Common Objects in Context — Tập dữ liệu phát hiện đối tượng |
| CRNN | Convolutional Recurrent Neural Network — Mạng nơ-ron tích chập hồi quy |
| CSV | Comma-Separated Values — Định dạng bảng phân tách bằng dấu phẩy |
| CTC | Connectionist Temporal Classification — Phân loại thời gian kết nối |
| cuDNN | CUDA Deep Neural Network library — Thư viện mạng nơ-ron sâu trên GPU NVIDIA |
| CUDA | Compute Unified Device Architecture — Nền tảng tính toán GPU NVIDIA |
| DBNet | Differentiable Binarization Network — Mạng nhị phân hóa khả vi |
| DDP | Distributed Data Parallel — Huấn luyện phân tán đa GPU |
| DETR | Detection Transformer — Mô hình phát hiện đối tượng dựa trên Transformer |
| DLA | Document Layout Analysis — Phân tích bố cục tài liệu |
| DLL | Dynamic Link Library — Thư viện liên kết động (Windows) |
| DPI | Dots Per Inch — Độ phân giải ảnh |
| F1 | F1-score — Chỉ số hài hòa Precision-Recall |
| FN | False Negative |
| FP | False Positive |
| FP16 | 16-bit Floating Point — Dấu phẩy động 16-bit |
| FPN | Feature Pyramid Network — Mạng kim tự tháp đặc trưng |
| GFLOPs | Giga Floating Point Operations Per Second — Tỷ phép tính dấu phẩy động mỗi giây |
| GPU | Graphics Processing Unit — Bộ xử lý đồ họa |
| GT | Ground Truth — Nhãn chuẩn |
| HTML | HyperText Markup Language — Ngôn ngữ đánh dấu siêu văn bản |
| ICDAR | International Conference on Document Analysis and Recognition — Hội nghị quốc tế về phân tích và nhận dạng tài liệu |
| IoU | Intersection over Union — Chỉ số giao-trên-hợp |
| JSON | JavaScript Object Notation — Định dạng trao đổi dữ liệu |
| LLM | Large Language Model — Mô hình ngôn ngữ lớn |
| LoRA | Low-Rank Adaptation — Kỹ thuật fine-tune tham số thấp |
| LSTM | Long Short-Term Memory — Mạng nhớ dài ngắn hạn |
| mAP | mean Average Precision — Độ chính xác trung bình tổng hợp |
| MIM | Masked Image Modeling — Mô hình hóa ảnh có che khuất |
| MLM | Masked Language Modeling — Mô hình hóa ngôn ngữ có che khuất |
| MLOps | Machine Learning Operations — Vận hành hệ thống học máy |
| NMS | Non-Maximum Suppression — Triệt tiêu cực đại không phải cực đại |
| OCR | Optical Character Recognition — Nhận dạng ký tự quang học |
| OOM | Out of Memory — Hết bộ nhớ |
| PDF | Portable Document Format — Định dạng tài liệu di động |
| PII | Personally Identifiable Information — Thông tin nhận dạng cá nhân |
| QLoRA | Quantized Low-Rank Adaptation — Thích nghi hạng thấp có lượng tử hóa |
| RAG | Retrieval-Augmented Generation — Sinh văn bản có tăng cường truy xuất |
| ReLU | Rectified Linear Unit — Hàm kích hoạt tuyến tính chỉnh lưu |
| ResNet | Residual Network — Mạng nơ-ron tàn dư |
| RNN | Recurrent Neural Network — Mạng nơ-ron hồi quy |
| SGD | Stochastic Gradient Descent — Hạ gradient ngẫu nhiên |
| SLA | Service Level Agreement — Thỏa thuận mức dịch vụ |
| SPPF | Spatial Pyramid Pooling — Fast |
| SOTA | State-of-the-Art — Tốt nhất hiện tại |
| TATR | Table Transformer — Mô hình nhận dạng cấu trúc bảng |
| TCO | Total Cost of Ownership — Tổng chi phí sở hữu |
| TP | True Positive |
| VFL | Varifocal Loss — Hàm mất mát tiêu điểm biến thiên |
| VGG | Visual Geometry Group — Kiến trúc CNN cổ điển |
| ViT | Vision Transformer — Transformer ứng dụng cho ảnh |
| VRAM | Video Random Access Memory — Bộ nhớ GPU |
| WER | Word Error Rate — Tỷ lệ lỗi từ |
| XML | eXtensible Markup Language — Ngôn ngữ đánh dấu mở rộng |
| YOLO | You Only Look Once — Kiến trúc phát hiện đối tượng thời gian thực |

---

# CHƯƠNG 1: GIỚI THIỆU {#chương-1}

## 1.1 Đặt vấn đề

Các hệ thống AI hiện đại — từ chatbot doanh nghiệp, hệ thống hỏi đáp tài liệu (RAG) đến nền tảng phân tích pháp lý — đều có chung một điểm nghẽn: tài liệu thực tế tồn tại dưới dạng PDF, không phải văn bản thuần túy mà AI có thể tiêu thụ trực tiếp. Để một mô hình ngôn ngữ lớn có thể trả lời câu hỏi từ báo cáo tài chính hay tra cứu điều khoản trong hợp đồng pháp lý, bước đầu tiên bắt buộc là chuyển đổi trang PDF thành dữ liệu có cấu trúc — xác định đâu là tiêu đề, đâu là bảng số liệu, đâu là nội dung chính, và thứ tự đọc đúng là gì. Đây chính là bài toán mà luận văn này giải quyết.

Tuy nhiên, PDF không phải là định dạng được thiết kế để trích xuất nội dung. Không giống với văn bản thuần túy, một trang PDF là một bức tranh bố cục hai chiều — nơi văn bản, bảng biểu, hình ảnh và công thức toán học được sắp xếp theo logic trực quan mà máy tính không thể hiểu một cách trực tiếp. Thậm chí với các tài liệu PDF gốc kỹ thuật số, thứ tự luồng văn bản trong file thường không phản ánh đúng thứ tự đọc thực tế của người dùng — đặc biệt với bố cục nhiều cột, bảng biểu phức tạp hay chú thích đan xen.

Các phương pháp trích xuất nội dung truyền thống như `pdfplumber` hay `PyMuPDF` chỉ có thể lấy văn bản thô theo luồng byte trong file, không nhận biết được cấu trúc ngữ nghĩa của trang. Trong khi đó, OCR toàn trang (full-page OCR) nhận toàn bộ ảnh trang rồi trả về văn bản liên tục — mất hoàn toàn thông tin về loại vùng (đây là tiêu đề hay chú thích?), vị trí tương đối (đoạn này thuộc cột nào?) và quan hệ cấu trúc (bảng này liên quan đến hình nào?).

Bài toán **Phân tích bố cục tài liệu** (Document Layout Analysis — DLA) ra đời nhằm giải quyết khoảng trống này: tự động phát hiện và phân loại các vùng nội dung trên trang tài liệu trước khi thực hiện bất kỳ bước xử lý nào tiếp theo. Kết quả của DLA — tập các vùng có nhãn và tọa độ — là nền tảng để xây dựng các hệ thống trích xuất thông tin, tìm kiếm ngữ nghĩa, chuyển đổi định dạng và phân tích tài liệu quy mô lớn.

## 1.2 Mục tiêu nghiên cứu

Luận văn này đặt mục tiêu xây dựng một pipeline end-to-end hoàn chỉnh cho bài toán phân tích bố cục và trích xuất cấu trúc ngữ nghĩa từ tài liệu PDF, bao gồm các thành phần:

1. **Phát hiện bố cục:** Huấn luyện mô hình phát hiện đối tượng YOLO trên tập dữ liệu DocLayNet để phân loại 11 lớp vùng bố cục tài liệu, có xử lý vấn đề mất cân bằng lớp nghiêm trọng (97:1).

2. **Nhận dạng văn bản:** Lựa chọn và tích hợp engine OCR phù hợp thông qua thực nghiệm so sánh định lượng, với chiến lược định tuyến khác nhau cho từng loại vùng nội dung.

3. **Sắp xếp thứ tự đọc:** Triển khai thuật toán XY-Cut tùy chỉnh phù hợp với bố cục tài liệu đa dạng.

4. **Làm giàu ngữ nghĩa:** Tích hợp mô hình ngôn ngữ lớn để cấu trúc hóa đầu ra thành JSON có schema chuẩn, bao gồm tóm tắt và từ khóa.

5. **Đánh giá định lượng:** Xây dựng framework đánh giá ba cấp độ (phát hiện khối, OCR, LLM) trên tập dữ liệu DocLayNet, cung cấp số liệu minh bạch cho từng module.

## 1.3 Phạm vi nghiên cứu

Luận văn tập trung vào các giới hạn sau:

- **Tập dữ liệu:** DocLayNet v1.2 — 80,863 trang PDF, 11 lớp bố cục, 6 loại tài liệu (Financial Report, Scientific Article, Laws & Regulations, Government Tender, Manual, Patent).
- **Ngôn ngữ tài liệu:** Chủ yếu tiếng Anh. Engine docTR hỗ trợ Latin script; không áp dụng trực tiếp cho tiếng Việt hay các ngôn ngữ non-Latin.
- **Loại đầu vào:** Tài liệu PDF có thể render thành ảnh (cả PDF gốc kỹ thuật số và PDF scan). Không xử lý PDF có bảo mật mật khẩu hay watermark dày đặc.
- **Phần cứng:** Huấn luyện trên Kaggle Notebook GPU Tesla T4 (15GB VRAM). Inference trong pipeline được tối ưu cho môi trường single GPU.

## 1.4 Đóng góp chính

Trong luận văn này, em xây dựng và đánh giá một pipeline kết hợp object detection, OCR và LLM để giải quyết bài toán phân tích bố cục và trích xuất cấu trúc ngữ nghĩa từ tài liệu PDF. Thay vì đề xuất một kiến trúc mạng nơ-ron hoàn toàn mới, em tập trung vào việc thiết kế hệ thống có kiểm chứng định lượng rõ ràng: đặt câu hỏi nghiên cứu cụ thể, tiến hành thực nghiệm đối chứng có nhóm kiểm soát, và phân tích lỗi theo từng module để hiểu rõ điểm mạnh và hạn chế của từng giai đoạn. Luận văn có bốn đóng góp chính:

**(1) Bằng chứng thực nghiệm cho chiến lược oversampling lớp hiếm:** Em thực hiện ablation study với ba thí nghiệm trên DocLayNet — bộ dữ liệu có mức mất cân bằng lớp lên đến 97:1 — và nhận thấy rằng oversampling lớp hiếm ở cấp độ trang cho kết quả ổn định hơn so với augmentation hình học mạnh (mosaic, mixup, flip tích cực) trên tài liệu có cấu trúc bố cục cố định. Kết quả đạt được: mAP@0.5 = **0.922** (tập kiểm định) và **0.909** (tập kiểm tra), toàn bộ 11 lớp đạt AP > 0.83 kể cả các lớp cực hiếm như Title (0.47% dữ liệu) và Footnote (0.60%).

**(2) Khảo sát thực nghiệm lựa chọn OCR engine cho pipeline block-level:** Em so sánh định lượng ba engine OCR phổ biến — docTR, EasyOCR và TrOCR — trên 50 mẫu DocLayNet với các chỉ số CER và WER. Quy mô 50 mẫu được chọn vì mục tiêu là so sánh tương đối giữa các engine trên cùng điều kiện, không phải đo hiệu suất tuyệt đối — kết quả kiểm tra độ nhất quán trên 5 mẫu ban đầu so với 50 mẫu cho thấy thứ hạng các engine không thay đổi (§3.3.2), xác nhận 50 mẫu đủ cho mục đích lựa chọn. Kết quả cho thấy docTR phù hợp nhất với CER = 0.427 (thấp nhất trong ba engine) và tốc độ xử lý chấp nhận được. CER ~0.12 trên văn bản thân trang xác nhận lựa chọn này phù hợp với yêu cầu thực tế của pipeline.

**(3) Pipeline end-to-end với chi phí vận hành thấp:** Hệ thống hoàn chỉnh YOLO → docTR → XY-Cut → Gemini đạt content recall 0.800, table token F1 0.870, schema parse rate 99.8%, với chi phí xử lý và vận hành thấp — ít hơn 4–16× so với các dịch vụ thương mại tương đương (AWS Textract, Azure Document Intelligence; chi tiết tại §4.8). LLM được tích hợp ở chế độ nhận văn bản OCR thay vì toàn bộ ảnh trang, giúp giảm đáng kể chi phí token mà vẫn giữ được chất lượng đầu ra.

**(4) Framework đánh giá ba cấp và phân tích lỗi theo module:** Em xây dựng phương pháp đánh giá tách biệt từng giai đoạn (Cấp 1: Detection F1; Cấp 2: OCR CER; Cấp 3: LLM metrics) trên 485 mẫu DocLayNet. Cách tiếp cận này giúp xác định cụ thể nguồn gốc của 20% content recall bị thất thoát — detection miss (~10–12%), OCR noise (~4–6%), reading order/LLM (~3–5%), annotation noise (~2–3%) — thay vì chỉ nhìn vào chỉ số tổng hợp cuối cùng.

## 1.5 Câu hỏi nghiên cứu

Bốn câu hỏi nghiên cứu sau định hướng thiết kế thực nghiệm và cách diễn giải kết quả trong luận văn:

**RQ1.** Với bài toán phát hiện bố cục tài liệu có mất cân bằng lớp nghiêm trọng (97:1), chiến lược oversampling lớp hiếm có cải thiện mAP nhất quán hơn augmentation hình học mạnh hay không?

**RQ2.** Trong ngữ cảnh trích xuất văn bản theo block từ tài liệu PDF kỹ thuật số tiếng Anh, OCR engine nào trong số docTR, EasyOCR và TrOCR đạt CER thấp nhất và phù hợp nhất để tích hợp vào pipeline?

**RQ3.** Pipeline lai YOLO+OCR+LLM có đạt trade-off tốt hơn phương án rule-based truyền thống (PyMuPDF/pdfplumber) về chất lượng trích xuất cấu trúc, chi phí vận hành và độ trễ xử lý hay không?

**RQ4.** Khi đánh giá pipeline end-to-end, các lỗi phân bổ như thế nào giữa bốn giai đoạn (detection, OCR, reading order, LLM)? Giai đoạn nào là nút thắt về chất lượng và giai đoạn nào là nút thắt về độ trễ?

Bốn câu hỏi này được trả lời lần lượt tại: §4.2–4.3 (RQ1), §4.4 (RQ2), §4.8 (RQ3) và §4.5.1 (RQ4).

## 1.6 Cấu trúc luận văn

Phần còn lại của luận văn được tổ chức như sau:

- **Chương 2 — Cơ sở lý thuyết:** Trình bày nền tảng lý thuyết của bài toán DLA, kiến trúc mô hình YOLO, các phương pháp OCR hiện đại, thuật toán sắp xếp thứ tự đọc và vai trò của mô hình ngôn ngữ lớn trong xử lý tài liệu.

- **Chương 3 — Phương pháp và triển khai:** Mô tả chi tiết kiến trúc pipeline bốn giai đoạn: (1) phát hiện bố cục bằng YOLOv11s (bao gồm ablation study 3 thí nghiệm), (2) nhận dạng văn bản OCR, (3) sắp xếp thứ tự đọc bằng XY-Cut tùy chỉnh, (4) làm giàu ngữ nghĩa bằng LLM.

- **Chương 4 — Kết quả thực nghiệm:** Trình bày và phân tích kết quả định lượng từ framework đánh giá ba cấp, bao gồm kết quả YOLO training, đánh giá end-to-end phát hiện khối, chất lượng OCR và hiệu suất giai đoạn LLM.

- **Chương 5 — Kết luận:** Tóm tắt kết quả đạt được, đối chiếu với mục tiêu ban đầu, nêu rõ hạn chế và đề xuất hướng phát triển tiếp theo.

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT {#chương-2}

## 2.1 Bài toán Phân tích Bố cục Tài liệu

### 2.1.1 Định nghĩa và phân loại bài toán

Phân tích bố cục tài liệu (Document Layout Analysis — DLA) là bài toán nhận dạng và phân loại các vùng nội dung trên trang tài liệu dạng ảnh hoặc PDF. Đầu vào của bài toán là ảnh trang tài liệu; đầu ra là tập các bounding box kèm nhãn lớp xác định loại vùng nội dung tương ứng — văn bản, tiêu đề, bảng biểu, hình ảnh, công thức, v.v.

DLA về bản chất là một bài toán phát hiện đối tượng (object detection) với miền dữ liệu đặc thù. Nhóm nghiên cứu của Pfitzmann [1] mô tả bài toán như sau: *"Document Layout Analysis aims to identify, group, and classify the regions of a page scan or PDF according to their semantic roles."* Điểm khác biệt quan trọng so với phát hiện đối tượng tổng quát là các đối tượng trong tài liệu tuân theo ngữ nghĩa có cấu trúc — cùng nhãn "Text" có thể là đoạn thân bài, chú thích ảnh, hay text trong bảng — mỗi trường hợp cần xử lý khác nhau trong pipeline hạ nguồn.

Bài toán DLA được phân thành hai hướng tiếp cận chính: *phân tích vùng* (region-level) xác định bounding box của từng vùng ngữ nghĩa, và *phân tích pixel* (pixel-level) gán nhãn cho từng điểm ảnh. Trong luận văn này, hướng tiếp cận region-level được chọn vì tương thích trực tiếp với kiến trúc object detection (YOLO) và thuận tiện cho pipeline xử lý downstream.

### 2.1.2 Thách thức đặc thù

Không giống với object detection trên ảnh tự nhiên (COCO), bài toán DLA đặt ra những thách thức riêng biệt về mặt dữ liệu và mô hình:

**Đa dạng bố cục liên miền:** Nhóm nghiên cứu của Pfitzmann [1] chỉ ra rằng các tài liệu thuộc các lĩnh vực khác nhau có quy ước bố cục hoàn toàn khác nhau. Báo cáo tài chính thường có bảng biểu dày đặc và số liệu nhiều cột; bài báo khoa học dùng bố cục hai cột với công thức toán; văn bản pháp lý có cấu trúc điều khoản đánh số phức tạp. Chính vì lý do này, DocLayNet [1] bao gồm 6 loại tài liệu khác nhau thay vì chỉ tập trung vào một domain như PubLayNet [2].

**Mất cân bằng lớp nghiêm trọng:** Trong DocLayNet, lớp Text chiếm tới 45.82% tổng annotation trong khi Title chỉ chiếm 0.47% — tỷ lệ chênh lệch gần 100:1. Theo Crasto [10], *"the class imbalance problem in object detection is twofold: foreground-background imbalance and foreground-foreground imbalance,"* và foreground-foreground imbalance ít được nghiên cứu hơn nhưng cũng không kém phần quan trọng trong bài toán tài liệu.

**Sự mơ hồ ranh giới annotation:** Nhóm của Pfitzmann [1] báo cáo rằng inter-annotator agreement của DocLayNet đạt mAP@0.5-0.95 = **82–83%** — đây đồng thời là trần chất lượng lý thuyết của bài toán. Ranh giới giữa Caption và Footnote, giữa Section-header và Title, không có định nghĩa tuyệt đối và phụ thuộc vào nhận định chủ quan của annotator.

**Đặc trưng hình học phi tự nhiên:** Phân tích DocLayNet (mục 3.2.1) cho thấy 74.2% bounding box có tỷ lệ chiều rộng/cao (Aspect Ratio) lớn hơn 5:1, với median AR = 11.67. Đây là hệ quả của bản chất dải văn bản ngang. Ngoài ra, 47.4% annotation thuộc nhóm "tiny object" (dưới 1% diện tích trang) — tỷ lệ cao hơn nhiều so với dataset ảnh tự nhiên như COCO (~10%).

### 2.1.3 Tập dữ liệu

**Bảng 2.1: So sánh các tập dữ liệu Document Layout Analysis tiêu biểu**

| Tập dữ liệu | Năm | Số trang | Số lớp | Loại tài liệu | Chú thích | Trần chất lượng |
|------------|-----|---------|--------|--------------|-----------|----------------|
| PubLayNet [2] | 2019 | 360,000+ | 5 | Scientific (PubMed) | Tự động (XML↔PDF) | Không báo cáo |
| DocBank [3] | 2020 | 500,000 | 12 | Scientific (arXiv) | Weak (LaTeX source) | Không báo cáo |
| DocLayNet [1] | 2022 | 80,863 | **11** | **6 loại đa dạng** | **Thủ công (chuyên gia)** | **82–83% mAP** |

Luận văn sử dụng **DocLayNet** vì ba lý do: (1) annotation thủ công bởi chuyên gia — không có noise từ XML/LaTeX mismatch; (2) 6 loại tài liệu đa dạng (tài chính, pháp lý, khoa học, patent, v.v.) phản ánh tình huống thực tế; (3) document-wise split tránh data leakage, đảm bảo kết quả đánh giá không bị thổi phồng. Inter-annotator agreement 82–83% mAP là trần chất lượng lý thuyết — mọi model đạt cao hơn mức này trên tập test đều có dấu hiệu overfit.

### 2.1.4 Các phương pháp SOTA trên DocLayNet

**Bảng 2.2: So sánh các phương pháp tiêu biểu trên DocLayNet**

| Phương pháp | Kiến trúc | mAP@0.5 | Tốc độ (FPS) | Cần OCR trước | Dữ liệu bổ sung |
|------------|---------|---------|------------|--------------|----------------|
| LayoutLMv3-L [7] | Transformer đa modal | 0.921 | ~2 | Có | Không |
| DiT-Cascade-L [8] | ViT-L + Cascade MRCNN | 0.928 | ~1.5 | Không | IIT-CDIP 42M |
| DocLayout-YOLO [6] | YOLOv10 + DocSynth-300K | 0.931 | ~22 | Không | DocSynth-300K |
| **YOLOv11s (luận văn)** | **CSP+C3k2** | **0.909** | **~82** | **Không** | **Không** |

Luận văn chọn YOLOv11s thay vì các SOTA Transformer (LayoutLMv3, DiT) vì hai lý do thực tế: tốc độ inference ~82 FPS phù hợp với pipeline xử lý hàng loạt, và không yêu cầu OCR engine chạy trước. Khoảng cách 2.2 điểm mAP so với SOTA được bù đắp bởi chiến lược dữ liệu (§3.2.2) — YOLOv11s + oversampling đạt 0.922 val mAP mà không cần dữ liệu tổng hợp bổ sung.

> Li và cộng sự [11] (RoDLA, CVPR 2024) chỉ ra LayoutLMv3 và DiT giảm hiệu suất đáng kể trên tài liệu nhiễu thực tế — YOLO duy trì tốt hơn trong điều kiện ngoài benchmark sạch.

## 2.2 Phát hiện đối tượng với YOLO

### 2.2.1 Kiến trúc YOLOv11s và lý do lựa chọn

YOLOv11 [4] là mô hình phát hiện đối tượng một giai đoạn (single-stage): dự đoán bounding box và class score trong một forward pass duy nhất, không có bước region proposal riêng biệt như Faster R-CNN. Ba cải tiến so với YOLOv8 liên quan trực tiếp đến bài toán tài liệu:

- **C3k2 Block:** Giảm 22% tham số so với YOLOv8m nhờ kernel nhỏ hơn, giữ nguyên receptive field qua CSP connection.
- **SPPF:** Pooling đa scale tuần tự — xử lý đồng thời Picture lớn (40% diện tích trang) và Caption nhỏ (vài pixel).
- **C2PSA:** Self-attention trong neck giúp model chú ý đúng vùng — quan trọng cho Footnote/Caption thường bị lấn át bởi Text block lớn gần đó.

**Lý do chọn variant YOLOv11s (9.4M tham số):** Môi trường Kaggle T4 (15GB VRAM, giới hạn 12 giờ/session) không cho phép variant lớn hơn — YOLOv11m chỉ đạt batch=12 (không ổn định) và cần ~14 giờ cho 50 epoch. YOLOv11s cho phép batch=24 và hoàn thành trong 8–9 giờ. Khoảng cách 4.5 điểm mAP trên COCO so với YOLOv11m được bù đắp bởi chiến lược oversampling — kết quả cuối đạt 0.922 val mAP.

### 2.2.2 Xử lý mất cân bằng lớp

Mất cân bằng lớp trong object detection xảy ra ở hai cấp độ: (1) **foreground–background imbalance** — số lượng anchor/proposal là background áp đảo so với positive; (2) **inter-class imbalance** — trong số positive, lớp phổ biến (Text) có thể nhiều hơn lớp hiếm (Footnote, Title) hàng chục lần.

YOLOv11 xử lý cấp độ thứ nhất thông qua **VFL (Varifocal Loss)** [9] — biến thể của Focal Loss vừa down-weight easy negatives vừa up-weight positive examples theo chất lượng localization (IoU score), giúp gradient tập trung vào các example khó mà không cần hard negative mining thủ công.

Tuy nhiên, VFL không đủ cho inter-class imbalance cực đoan như DocLayNet (97:1). Nhóm của Crasto [10] chỉ ra rằng với YOLO-based detector, instance-level oversampling và loss reweighting không cải thiện đáng kể lớp hiếm — do anchor matching đã lọc background sẵn, nên thêm weighting không bổ sung nhiều. Ngược lại, **page-level oversampling** — lặp lại toàn bộ trang chứa lớp hiếm thay vì nhân bản từng bbox riêng lẻ — giữ nguyên spatial context đầy đủ, giúp model học được spatial prior của từng lớp (Footnote luôn ở góc dưới, Caption luôn ngay dưới Figure). Chiến lược này và kết quả ablation được trình bày chi tiết tại §3.2.2–3.2.3.

### 2.2.4 Thách thức đặc thù của DLA với YOLO

Hai đặc trưng hình học của DocLayNet đặt ra thách thức cụ thể (phân tích chi tiết tại §3.2.1):

- **Aspect ratio cực lớn:** Median AR = 11.67 — text block điển hình rộng gần 12× chiều cao, trái ngược với object tự nhiên (AR 0.5–2.0). Kiến trúc anchor-free của YOLOv11 linh hoạt hơn anchor-based với extreme AR, nhưng cần nhiều epoch để hội tụ đúng range.
- **Tiny object chiếm 47.4%:** Hầu hết Footnote, Caption, Formula có diện tích dưới 1% trang. Luận văn giải quyết bằng page-level oversampling kết hợp imgsz=640 — đảm bảo feature map stride-32 vẫn đủ resolution cho tiny object mà không cần module chuyên biệt như DocLayout-YOLO.

YOLO được khởi tạo từ COCO pretrained weights: backbone giữ low-level features (edge, texture), detection head khởi tạo lại vì 11 lớp tài liệu khác hoàn toàn 80 lớp COCO. Kết quả: `pretrained=True` đạt mAP@0.5=0.922 sau 50 epoch — hội tụ nhanh nhờ backbone có sẵn.

## 2.3 Nhận dạng văn bản (OCR)

### 2.3.1 docTR và kiến trúc OCR hai giai đoạn

OCR hiện đại gồm hai giai đoạn: **text detection** (DBNet [21] — học threshold map thay vì threshold cố định, train end-to-end) và **text recognition** (CRNN [22] — CNN trích feature → BiLSTM học chuỗi → CTC decode; hoặc ViTSTR dùng ViT encoder, chính xác hơn nhưng chậm hơn).

**docTR** (Mindee) đóng gói cả hai vào một unified API, trả về kết quả phân cấp page → block → line → word kèm confidence score. Output phân cấp này tương thích trực tiếp với block-level từ YOLO; confidence per word cho phép trigger Gemini Vision fallback khi confidence < 0.70 cho bảng biểu và văn bản khó đọc.

## 2.4 Sắp xếp thứ tự đọc

### 2.4.1 XY-Cut và biến thể

Reading order xác định thứ tự đọc các block sau detection — sai thứ tự tạo ra văn bản vô nghĩa khi đưa vào LLM, ảnh hưởng trực tiếp đến chất lượng extraction.

**XY-Cut [12]** (Pavlidis & Zhou, 1992) dùng projection profile: chiếu bbox lên trục X/Y, tìm vùng trống để cắt đệ quy trang thành các strip → column → block, sau đó DFS traversal cho thứ tự top-down, left-right. Giới hạn: không xử lý được hình ảnh span nhiều cột và layout tự do.

**XY-Cut++ [13]** (Liu, arXiv 2025) bổ sung pre-mask cho Table/Figure trước khi cắt và phân tích đa cấp (Page → Column → Block → Line), đạt 98.8 BLEU so với ~82.0 BLEU của XY-Cut gốc. Tuy nhiên, XY-Cut++ yêu cầu pixel-level mask segmentation — không có sẵn từ YOLO bbox. Đây là lý do luận văn thiết kế thuật toán XY-Cut tùy chỉnh riêng trực tiếp trên bbox (§3.4).

## 2.5 Mô hình ngôn ngữ lớn trong xử lý tài liệu

### 2.5.1 Vai trò LLM và lựa chọn Gemini 2.5 Flash

LLM đảm nhiệm hai vai trò trong pipeline: (1) **cấu trúc hóa ngữ nghĩa** — phân tích text blocks theo reading order để map sang schema JSON (heading, body, table, formula); (2) **sửa lỗi OCR theo ngữ cảnh** — suy luận từ văn bản xung quanh để correction ký tự sai, nhất là với dấu phụ [14]. Cả hai đều không thể thực hiện bằng rule-based thuần túy.

**Gemini 2.5 Flash [24]** được chọn vì ba đặc điểm phù hợp với pipeline:
- **Context window 1M token:** Trang tài liệu dày đặc nhất cũng chỉ ~5,000 token — không bao giờ phải truncate.
- **Vision API:** Nhận ảnh crop bảng/công thức, trả về HTML table hoặc LaTeX — OCR truyền thống không xử lý được cấu trúc này.
- **Thinking budget tắt được:** Thực nghiệm (§3.5) cho thấy tắt thinking giảm chi phí 4.7× ($0.018 → $0.0038/trang) trong khi schema parse rate giữ nguyên 99.8% — cấu trúc hóa theo schema cố định không cần chain-of-thought.

Hai giới hạn thực tế của LLM trong pipeline được phân tích chi tiết tại §5.5: hallucination khi OCR input nhiễu nặng, và hiệu quả correction phụ thuộc ngôn ngữ [15].

---

# CHƯƠNG 3: PHƯƠNG PHÁP VÀ TRIỂN KHAI {#chương-3}

## 3.1 Tổng quan kiến trúc pipeline

Pipeline được xây dựng theo kiến trúc bốn giai đoạn nối tiếp, xử lý từng trang tài liệu PDF một cách độc lập. Hình 3.1 mô tả luồng xử lý tổng thể.

**Hình 3.1: Kiến trúc tổng thể pipeline bốn giai đoạn**

![Hình 3.1: Kiến trúc tổng thể pipeline bốn giai đoạn](thesis_figures/fig_3_1_pipeline.png)

**Tiền xử lý:** Mỗi trang PDF được render thành ảnh PNG ở DPI=150 bằng thư viện `pdf2image`. Ảnh sau đó được thêm padding trắng để đạt kích thước 1025×1025 (letterbox padding — giữ nguyên tỷ lệ khung hình gốc của trang). Resolution DPI=150 là điểm cân bằng giữa chất lượng đủ để OCR và tốc độ xử lý hợp lý.

**Quản lý tài nguyên:** YOLO (YOLOv11s, FP16) và docTR được load/unload tuần tự trên GPU T4 16GB để tránh OOM. Gemini API call bất đồng bộ, không chiếm GPU.

## 3.2 Giai đoạn 1: Phát hiện bố cục (YOLOv11s)

### 3.2.1 Phân tích phân phối dữ liệu DocLayNet

Trước khi thiết kế chiến lược huấn luyện, phân tích chi tiết phân phối dữ liệu DocLayNet được thực hiện trên toàn bộ 941,123 annotation của 69,375 ảnh tập train.

**Bảng 3.1: Phân phối annotation 11 lớp trong DocLayNet (tập train)**

| STT | Lớp | Số lượng | Tỷ lệ (%) | Nhóm |
|-----|-----|---------|----------|------|
| 1 | Text | 431,251 | 45.82 | Dominant |
| 2 | List-item | 161,818 | 17.19 | Dominant |
| 3 | Section-header | 118,590 | 12.60 | Large |
| 4 | Page-footer | 61,313 | 6.51 | Medium |
| 5 | Page-header | 47,973 | 5.10 | Medium |
| 6 | Picture | 39,667 | 4.21 | Medium |
| 7 | Table | 30,070 | 3.20 | Medium |
| 8 | Formula | 21,167 | 2.25 | Small |
| 9 | Caption | 19,218 | 2.04 | Small |
| 10 | Footnote | 5,619 | 0.60 | Rare |
| 11 | Title | 4,437 | 0.47 | Rare |
| | **Tổng** | **941,123** | **100** | |

Phân tích cho thấy **mức độ mất cân bằng 97:1** giữa lớp nhiều nhất (Text, 431K) và ít nhất (Title, 4.4K). Đây là mức chênh lệch gần 5 lần so với ước lượng ban đầu (20:1), đặt ra yêu cầu phải có chiến lược xử lý class imbalance chủ động.

![Hình 3.2: Phân phối 11 lớp annotation DocLayNet](thesis_figures/exist_class_distribution_full.png)

**Hình 3.2: Phân phối 11 lớp annotation DocLayNet — train/val/test**

**Bảng 3.2: Thống kê kích thước và tỷ lệ khung hình bounding box**

| Đặc trưng | Giá trị |
|---------|---------|
| Tiny bbox (<1% diện tích ảnh) | 47.4% tổng annotation |
| Bbox có AR > 5:1 | 74.2% |
| Median Aspect Ratio | 11.67 |
| Mean Aspect Ratio | 15.39 |
| Số vật thể trung bình/trang | 13.6 |
| P95 số vật thể/trang | 28 |

Đặc điểm 74.2% bbox có tỷ lệ chiều rộng/cao > 5:1 phản ánh bản chất của text block tài liệu: rất rộng theo chiều ngang nhưng rất thấp theo chiều dọc. Phát hiện này dẫn đến quyết định bật chế độ batch hình chữ nhật trong cấu hình training để YOLO không bị méo bbox khi ghép ảnh thành batch.

![Hình 3.3a: Phân phối AR](thesis_figures/exist_aspect_ratio_dist.png) ![Hình 3.3b: Phân phối kích thước bbox](thesis_figures/exist_bbox_scale_dist.png)

**Hình 3.3: Phân phối tỷ lệ khung hình (trái) và kích thước bounding box (phải) trong DocLayNet**

### 3.2.2 Cấu hình huấn luyện

Dựa trên phân tích dữ liệu và ràng buộc phần cứng, cấu hình huấn luyện được xác định như sau. Môi trường Kaggle Notebook GPU T4 (15GB VRAM, giới hạn session 12 giờ) là yếu tố quyết định nhiều lựa chọn: YOLOv11s được chọn thay vì variant lớn hơn vì cho phép batch=24 ổn định trong giới hạn VRAM, đồng thời hoàn thành 50 epoch trong ~8–9 giờ mà không vượt session limit. imgsz=640 là điểm cân bằng giữa chất lượng phát hiện và tốc độ — tăng lên 1024 sẽ giảm batch xuống còn 6–8, làm mất ổn định SGD và kéo dài thời gian training vượt giới hạn Kaggle.

**Bảng 3.3: Cấu hình huấn luyện cố định giữa các thí nghiệm**

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| model | yolo11s | 9.4M params; batch=24 vừa T4 15GB; m/l variant OOM hoặc vượt session limit 12h |
| imgsz | 640 | Cho phép batch=24 trên T4 |
| batch | 24 | Tối đa trước ngưỡng OOM |
| optimizer | SGD | Ổn định hơn Adam với YOLO |
| lr0 | 0.01 | Learning rate khởi tạo chuẩn |
| cos_lr | True | Cosine annealing schedule |
| epochs | 50 | Đủ dài, có early stopping |
| patience | 15 | Early stopping patience |
| seed | 42 | Reproducibility |
| nbs | 64 | Effective batch qua gradient accumulation |
| pretrained | True | COCO pretrained weights |
| rect | True | Xử lý bbox AR cực lớn |

**Chiến lược oversampling cho lớp hiếm:** Title ×3 (4.4K → 13.3K effective), Footnote ×3 (5.6K → 16.8K), Caption ×2 (19.2K → 38.4K). Formula (21K) không oversample vì đã ở mức trung bình. Triển khai bằng cách nhân bản đường dẫn ảnh trong file `dataset.yaml` — mỗi lần load sẽ áp dụng augmentation khác nhau nên không bị overfit hoàn toàn.

**Confidence threshold per-class (inference):** Mặc định 0.25, override: Picture=0.50 (giảm false positive cho hình ảnh), Section-header=0.40, Caption=0.40, Table=0.12 (recall-oriented cho bảng biểu quan trọng), Formula=0.45 (giảm false positive với ký tự đặc biệt), Footnote=0.35 (tăng recall cho footnote ngắn ở cuối trang).

### 3.2.3 Thiết kế ablation study

Để xác định chiến lược tốt nhất, ba thí nghiệm ablation được thiết kế với nguyên tắc thay đổi duy nhất một yếu tố mỗi lần. Ba thí nghiệm chỉ khác nhau ở hai khía cạnh: chiến lược dữ liệu (có oversampling hay không) và cường độ augmentation — các tham số còn lại giữ cố định để đảm bảo tính kiểm soát.

**Bảng 3.6: Các tham số khác nhau giữa 3 thí nghiệm ablation** (tham số giống nhau xem Bảng 3.3)

| Tham số | TN1 — Baseline | TN2 — Oversampling | TN3 — Aug. mạnh |
|---------|:--------------:|:------------------:|:---------------:|
| Dataset | DocLayNet gốc | DocLayNet + oversampling | DocLayNet gốc |
| Oversampling | — | Title ×3, Footnote ×3, Caption ×2 | — |
| fliplr | 0.0 | 0.0 | **0.5** |
| scale | 0.5 | 0.5 | **0.4** |
| close\_mosaic | 10 | 10 | **20** |

**Hình 3.4: Sơ đồ thiết kế ablation study — 3 thí nghiệm**

![Hình 3.4: Sơ đồ thiết kế ablation study](thesis_figures/fig_3_4_ablation.png)

**Thí nghiệm 1 — Baseline:** Cấu hình bảo thủ, không can thiệp vào phân phối dữ liệu. Tắt lật ngang (`fliplr=0.0`) vì tài liệu có cấu trúc không gian nhất quán — lật ngang sẽ đặt footnote (vốn ở góc dưới trái) sang bên phải, vi phạm prior bố cục tài liệu.

**Thí nghiệm 2 — Oversampling:** Giữ nguyên augmentation của Baseline, chỉ thêm oversampling ở cấp độ trang. Mỗi trang chứa ít nhất một annotation Title được lặp lại 3 lần trong danh sách training, tương tự cho Footnote và Caption. Cách tiếp cận này tăng tần suất xuất hiện các lớp hiếm mà không thay đổi phân phối trong từng ảnh đơn lẻ.

**Thí nghiệm 3 — Augmentation mạnh:** Bật lật ngang (`fliplr=0.5`) và tắt mosaic muộn hơn (`close_mosaic=20`) để model học feature thuần hơn ở cuối quá trình huấn luyện. Mục tiêu kiểm chứng xem augmentation mạnh có thay thế được oversampling không — kết quả cho thấy không, đặc biệt Footnote AP giảm mạnh nhất khi bật flip do vi phạm prior bố cục nêu trên.

### 3.2.4 Khó khăn kỹ thuật trong quá trình huấn luyện

Trong quá trình thực hiện ablation study trên Kaggle T4, nhiều thách thức kỹ thuật phát sinh:

**(1) Tràn bộ nhớ GPU (OOM):** Cấu hình yolo11s + imgsz=640 + batch=32 gây OOM do các batch chứa trang có nhiều annotation lớn. Giải pháp là giảm batch xuống 24 và dùng `nbs=64` để bù qua gradient accumulation — đạt effective batch size 64 mà không cần VRAM tương ứng.

**(2) Multi-GPU thất bại:** Thử nghiệm DDP (Distributed Data Parallel) trên 2×T4 tốn thêm ~5GB VRAM/GPU cho synchronization overhead, gây OOM. Quyết định cuối cùng là single GPU + gradient accumulation.

**(3) Session timeout Kaggle:** Kaggle giới hạn 12 giờ/session. Sử dụng `save_period=5` để lưu checkpoint mỗi 5 epoch và `resume=True` để tiếp tục ở session mới. Thực tế mỗi thí nghiệm cần 2 session (~9 giờ/thí nghiệm tổng cộng).

## 3.3 Giai đoạn 2: Nhận dạng văn bản

### 3.3.1 PaddleOCR — Lựa chọn ban đầu và lý do thay thế

PaddleOCR (Baidu PaddlePaddle) được lựa chọn ban đầu do dẫn đầu nhiều leaderboard OCR đa ngôn ngữ tại thời điểm khảo sát (tháng 11/2024). Tuy nhiên, quá trình tích hợp vào pipeline PyTorch trên Windows 11 + CUDA 12.1 gặp ba vấn đề liên tiếp không thể giải quyết trong phạm vi thời gian luận văn.

Vấn đề đầu tiên phát hiện được là một silent accuracy bug: `PaddleOCR(lang='en')` tự động fallback về model tiếng Trung khi model tiếng Anh chưa được cache, khiến CER đo được là ~18% thay vì ~5% kỳ vọng. Pin version thư viện giảm CER xuống ~12% nhưng vẫn cao hơn docTR.

Vấn đề thứ hai là xung đột phiên bản cuDNN: Paddle yêu cầu cuDNN 8.6 trong khi PyTorch 2.1 cần cuDNN 8.9 — hai thư viện không thể coexist trong cùng môi trường. Các phương án khắc phục (conda env riêng, subprocess IPC) đều thất bại.

Vấn đề thứ ba và nghiêm trọng nhất là DLL collision trên Windows: `paddle_inference.dll` và `torch_cuda.dll` xung đột trong cùng CUDA context, gây crash không tái hiện được với tỷ lệ ~30–40%. Không tìm được giải pháp ổn định.

Nguyên nhân gốc rễ là PaddlePaddle duy trì CUDA runtime riêng (static-link cuDNN cụ thể), về bản chất không thể coexist với PyTorch trong cùng process. Bên cạnh đó, benchmark SOTA của PaddleOCR được đo trên scene text (ICDAR), không phải tài liệu in PDF — CER thực tế trên DocLayNet không vượt trội engine PyTorch-native. Quyết định: **thay thế hoàn toàn bằng engine PyTorch-native** và benchmark lại theo điều kiện kiểm soát.


### 3.3.2 Benchmark lựa chọn OCR engine

Ba engine được đánh giá trên 50 mẫu DocLayNet được chọn ngẫu nhiên, sử dụng văn bản Ground Truth từ text layer PDF gốc làm chuẩn.

**Bảng 3.4: Kết quả benchmark 3 engine OCR trên 50 mẫu DocLayNet**

| Engine | CER ↓ | WER ↓ | Text Coverage ↑ | Tốc độ (s/block) | Ghi chú |
|--------|-------|-------|----------------|-----------------|---------|
| **docTR** | **0.427** | **0.548** | 73.5% | 0.144 | ✅ Tốt nhất tổng thể |
| EasyOCR | 0.448 | 0.615 | 74.4% | 0.130 | Chậm hơn trên GPU |
| TrOCR [23] | 0.706 | 0.836 | 98.0% | 0.236 | ❌ Loại — max_length=21 tokens |

TrOCR bị loại do giới hạn cứng `max_length=21 tokens` trong kiến trúc Transformer decoder — bất kỳ dòng văn bản nào dài hơn 21 từ đều bị cắt ngắn, dẫn đến WER = 0.836 dù coverage cao. docTR được chọn vì CER thấp nhất (0.427) và tốc độ chấp nhận được (0.144s/block).

**Tính nhất quán qua quy mô mẫu:** Benchmark bổ sung so sánh kết quả 5 mẫu ban đầu (thử nghiệm sơ bộ) với 50 mẫu đầy đủ:

| Engine | CER (5 mẫu) | CER (50 mẫu) | Thay đổi |
|--------|------------|-------------|---------|
| docTR | 0.397 | 0.427 | +0.030 |
| EasyOCR | 0.443 | 0.448 | +0.005 |
| TrOCR | 0.693 | 0.706 | +0.013 |

Thứ hạng các engine không thay đổi khi tăng từ 5 lên 50 mẫu, xác nhận tính ổn định của kết quả benchmark. CER của docTR tăng nhẹ (+0.030) khi mở rộng mẫu — phản ánh sự xuất hiện của các trang khó hơn (font nhỏ, ký hiệu đặc biệt) trong tập lớn hơn.

**Phân tích trường hợp EasyOCR vượt trội docTR:** Trong 50 mẫu, EasyOCR cho CER thấp hơn docTR ở 13/50 mẫu (26%), chủ yếu trên tài liệu có font không chuẩn hoặc cỡ chữ rất nhỏ đều đặn. Biên độ chênh lệch trung bình khi EasyOCR thắng: ~0.05 CER. Khi docTR thắng (37/50 mẫu, 74%): biên độ trung bình ~0.09 CER — docTR thắng với biên độ lớn hơn gần 2×. Do đó, docTR là lựa chọn tốt hơn về tổng thể dù không thống trị tuyệt đối.

**Diễn giải CER = 0.427:** Con số CER tổng thể 0.427 có thể gây hiểu nhầm nếu không phân tích theo lớp. Benchmark 50 mẫu gồm tất cả loại khối — kể cả Page-footer và Page-header là các lớp có bố cục bất thường (số trang, tiêu đề đầu trang ngắn, ký tự đặc biệt) đẩy CER lên cao. Thêm vào đó, 5/50 mẫu có block kích thước cực nhỏ hoặc nội dung ký hiệu đặc biệt khiến tất cả engine đều cho CER = 1.0; loại trừ 5 mẫu này, CER thực tế của docTR giảm xuống ~0.38. Khi phân tách theo lớp trong đánh giá đầy đủ (Bảng 4.6), CER của các lớp văn bản chính — Text và List-item — chỉ là **~0.12**, tức gần 88% ký tự nhận dạng đúng — mức chấp nhận được cho tài liệu PDF Latin-script gốc kỹ thuật số. Do đó, CER = 0.427 phản ánh đặc điểm phân bố dữ liệu không đồng đều trong benchmark chứ không phải hiệu suất thực tế trên nội dung chính của tài liệu.

### 3.3.3 Chiến lược định tuyến OCR

Không phải mọi loại khối đều phù hợp với OCR truyền thống. Chiến lược định tuyến theo nhãn YOLO như sau:

**Hình 3.6: Sơ đồ định tuyến OCR theo nhãn khối**

![Hình 3.6: Sơ đồ định tuyến OCR](thesis_figures/fig_3_6_ocr_routing.png)

**Tiền xử lý ảnh crop trước khi đưa vào docTR:** Crop theo tọa độ YOLO bbox + padding 6px mỗi chiều (để tránh mất ký tự ở rìa). Scale thích ứng theo kích thước block: nếu chiều cao crop nhỏ hơn 96px (ngưỡng target), scale lên để đạt mức đó; nếu crop đã đủ lớn, scale = 1.0 (không upscale lãng phí). Giới hạn trên là 3× để tránh tiêu tốn VRAM với block nhỏ cực đoan. Chiều nhỏ nhất sau scale tối thiểu 32px. Ảnh crop được chuyển sang RGB trước khi đưa vào docTR. Cách tiếp cận scale thích ứng này thay thế scale cố định 2× trước đó, giúp xử lý đúng cả block lớn (Title, Text dài) lẫn block nhỏ (Footnote, Page-footer).

## 3.4 Giai đoạn 3: Sắp xếp thứ tự đọc (XY-Cut tùy chỉnh)

### 3.4.1 So sánh các phương pháp

**Bảng 3.5: So sánh thuật toán reading order**

| Khía cạnh | XY-Cut cổ điển [12] | XY-Cut++ [13] | Triển khai trong luận văn |
|-----------|-------------------|--------------|--------------------------|
| Năm | 2005 | 2025 | 2025 |
| Cơ chế phân tách | Projection profile, valley | Hierarchical mask | X-projection gap + width ≥ 45% |
| Multi-column | Đệ quy đơn giản | Mask-based column grouping | Group theo row overlap, ratio ≥ 0.25 |
| Full-width block | Không phân biệt | Phát hiện qua mask | Tách riêng, sort theo y1 |
| Đầu vào yêu cầu | Ảnh nhị phân | Mask segmentation | YOLO bbox (x,y,w,h) |
| Độ phức tạp | O(n log n) | O(n²) worst case | O(n log n) |
| Phù hợp với | Layout đơn giản | Layout phức tạp | Layout 1–2 cột chuẩn |

Lý do không dùng XY-Cut++ nguyên bản: thuật toán yêu cầu mask segmentation đầu vào (pixel-level binary mask của từng vùng), trong khi YOLO chỉ cung cấp bounding box. Tạo mask từ bbox là heuristic không chính xác và tăng thêm độ phức tạp không cần thiết.

### 3.4.2 Thuật toán XY-Cut tùy chỉnh

**Hình 3.7: Thuật toán XY-Cut tùy chỉnh — 4 bước**

![Hình 3.7: Thuật toán XY-Cut tùy chỉnh](thesis_figures/fig_3_7_xycut.png)

Thuật toán nhận đầu vào danh sách `LayoutBlock` với bbox `(x1, y1, x2, y2)` từ YOLO, trả về danh sách đã gán `reading_order` từ 0 đến n−1, gồm ba bước:

**Bước 1 — Phát hiện số cột:** Tách block full-width (rộng ≥ 45% trang, lề trái ≤ 15%) khỏi nhóm columnar. Với các block columnar, dựng X-projection coverage, tìm gap gần giữa trang nhất. Trang được phán đoán là 2 cột khi gap thực sự tồn tại và tỉ lệ `min/max ≥ 0.25` (chấp nhận lệch đến 1:4 giữa hai cột).

**Bước 2 — Sắp xếp theo bố cục:** Với 2 cột, phân block theo tâm x so với ranh giới cột, sort mỗi cột độc lập theo `(y1, x1)`, đọc hết cột trái rồi sang phải. Với 1 cột, gom block thành các hàng ngang theo vertical overlap, sort trong hàng theo `x1`, nối tuần tự từ trên xuống.

**Bước 3 — Gán chỉ số:** Mỗi block được gán `reading_order = i` theo vị trí trong danh sách kết quả, dùng để sắp xếp text trước khi đưa vào LLM.

**Độ phức tạp:** O(n log n) — chi phí chủ yếu là sorting. Với n ≤ 30 block/trang (trung bình 15.2 block theo benchmark), thuật toán chạy dưới 1ms — không đáng kể so với OCR (1.9s) và LLM (5.9s).

**Giới hạn của thuật toán:** Thuật toán hoạt động tốt với bố cục 1–2 cột chuẩn. Với layout báo nhiều cột không đều (3–4 cột với độ rộng khác nhau) hay text chạy vòng quanh hình ảnh, thứ tự đọc có thể sai. Đây là hướng cải thiện được đề xuất trong Chương 5.

## 3.5 Giai đoạn 4: Làm giàu ngữ nghĩa (Mô hình ngôn ngữ lớn)

Sau khi có danh sách khối theo thứ tự đọc (output của giai đoạn 3), toàn bộ nội dung trang được gửi trong một API call duy nhất đến mô hình ngôn ngữ lớn được chọn.

**Thiết kế prompt:** Prompt bao gồm (1) nội dung văn bản đã OCR của từng khối kèm nhãn loại, (2) yêu cầu schema JSON đầu ra cụ thể với các trường `sections`, `paragraphs`, `tables`, `summary`, `keywords`, `metadata`, và (3) hướng dẫn xử lý đặc biệt cho bảng (đã có HTML từ Gemini Vision) và công thức (đã có LaTeX).

**Schema JSON đầu ra:** Gồm trường `structured_json` (title, page_header, page_footer, sections với heading/paragraphs/footnotes/tables/formulas/figures), `summary` (3–5 câu), `keywords`, `page_type` và `language`. `page_header`/`page_footer` được tách ra khỏi `sections` vì là metadata cấp trang; `footnotes` tách khỏi `paragraphs` để downstream xử lý riêng. Schema đầy đủ tại Phụ lục E.

**Thinking mode:** Chế độ suy luận nội tại được tắt hoàn toàn. Thực nghiệm cho thấy việc tắt thinking mode giảm chi phí 4.7× (từ ~$0.018/trang xuống ~$0.0038/trang) trong khi chất lượng schema parse rate không thay đổi đáng kể (99.8% vs 99.8%) với bài toán cấu trúc hóa tài liệu có schema cố định.

**Xử lý lỗi:** Schema được validate bằng Pydantic. Nếu output không hợp lệ, hệ thống retry tối đa 2 lần với prompt bổ sung hint về lỗi. Tỷ lệ parse thành công sau retry: 99.8% (485/485 mẫu chỉ 1 fail). Ba metric chính đánh giá giai đoạn LLM — *schema parse rate* (tỷ lệ output hợp lệ schema), *content recall* (tỷ lệ token nội dung GT xuất hiện trong output, đo bằng fuzzy matching), và *table token F1* (F1 ở cấp token cho nội dung bảng) — được định nghĩa chi tiết trong §4.5.

### 3.5.1 Ablation giai đoạn LLM

Để đưa ra các quyết định thiết kế có cơ sở, ba thí nghiệm ablation được thực hiện cho giai đoạn LLM, tập trung vào hai biến số có ảnh hưởng lớn nhất đến cost và quality: **prompt verbosity** và **thinking mode**.

**Thí nghiệm A — Prompt verbosity (50 mẫu ngẫu nhiên):**

| Phiên bản prompt | Input tokens TB | Schema parse rate | Ghi chú |
|-----------------|----------------|-------------------|---------|
| v1 — Verbose (~1,800 tokens) | ~1,800 | ~94% | Mô tả chi tiết từng trường, nhiều ví dụ |
| v2 — Compact (~350 tokens) | ~350 | **99.8%** | Loại bỏ ví dụ thừa, giữ schema cốt lõi |

Prompt dài hơn không cải thiện parse rate — ngược lại, schema parse rate của v2 cao hơn vì prompt ngắn gọn giảm xác suất LLM "lạc đề" vào phần giải thích ví dụ. Input tokens giảm ~80%, tương đương tiết kiệm ~$0.00045/trang chi phí input.

**Thí nghiệm B — Thinking mode (50 mẫu):**

| Cấu hình | Chi phí/trang | Schema parse rate | Content recall |
|----------|--------------|-------------------|----------------|
| Thinking ON | ~$0.018 | 99.8% | tương đương |
| Thinking OFF | **~$0.0038** | 99.8% | tương đương |

Thinking mode được thiết kế cho bài toán suy luận đa bước phức tạp (toán học, lập luận logic). Với bài toán cấu trúc hóa tài liệu — về cơ bản là mapping văn bản vào schema JSON cố định — không có lợi ích đo lường được từ thinking, nhưng chi phí tăng 4.7×. Quyết định tắt thinking là hợp lý cho use case này.

**Thí nghiệm C — Lựa chọn model (phân tích chi phí–chất lượng):**

So sánh đầy đủ các model không được thực hiện do chi phí, nhưng phân tích lý thuyết dựa trên pricing và benchmark công khai:

| Model | Chi phí ước tính/trang | Lý do không chọn |
|-------|----------------------|-----------------|
| Gemini 2.5 Flash (chọn) | **~$0.0038** | Cost tối ưu, schema parse rate 99.8% |
| Gemini 2.5 Flash Thinking | ~$0.018 | Cost cao 4.7×, không cải thiện quality |
| Gemini 2.5 Pro | ~$0.060–0.080 | Cost cao 15–20×, overkill cho schema cố định |
| Llama-3.2-3B (local) | ~$0 (compute) | Chất lượng JSON thấp hơn, cần GPU riêng |

Với bài toán structured output có schema cố định và input chủ yếu là văn bản ngắn (<2,000 tokens/trang), Flash là điểm tối ưu trên đường cong cost–quality. So sánh thực nghiệm đầy đủ với local LLM là hướng nghiên cứu tiếp theo (xem Chương 5).

## 3.6 Minh họa kết quả pipeline

Phần này trình bày kết quả trực quan của pipeline qua từng giai đoạn trên một trang báo cáo tài chính thực tế từ tập DocLayNet, giúp hiểu rõ cách các module nối tiếp nhau để biến ảnh PDF thành dữ liệu có cấu trúc.

**Giai đoạn 1 — Phát hiện bố cục:**

![Hình 3.8: Kết quả YOLO detection trực quan](thesis_figures/fig1_yolo_detection.png)

**Hình 3.8: Kết quả phát hiện bố cục YOLOv11s trên tài liệu thực tế — bounding box màu theo lớp, số thứ tự đọc**

Hình 3.8 cho thấy YOLOv11s khoanh vùng chính xác 13 vùng nội dung trên trang, bao gồm các lớp đa dạng: Text (xanh lam), Table (đỏ), Section-header (cam), Page-footer (xám). Đáng chú ý là mô hình phân biệt đúng ranh giới giữa Table và các Text block liền kề — điều quan trọng để giai đoạn OCR xử lý từng vùng theo đúng phương thức. Số thứ tự trên mỗi box phản ánh reading order do thuật toán XY-Cut xác định: cột trái được đọc trước cột phải, Section-header đứng trước Text block bên dưới.

**Tổng quan bốn giai đoạn:**

![Hình 3.9: Bốn giai đoạn pipeline](thesis_figures/fig4_pipeline_stages.png)

**Hình 3.9: Minh họa bốn giai đoạn pipeline trên trang PDF mẫu — (a) ảnh gốc, (b) YOLO detection, (c) văn bản OCR theo block, (d) JSON cấu trúc đầu ra**

Hình 3.9 minh họa luồng xử lý từ đầu đến cuối: **(a)** ảnh PDF gốc chưa qua xử lý — máy tính chỉ thấy ma trận pixel; **(b)** sau YOLO, trang được chia thành các vùng có nhãn và tọa độ rõ ràng; **(c)** docTR nhận từng crop riêng lẻ và trả về văn bản theo block, đã được sắp xếp theo reading order từ XY-Cut; **(d)** Gemini nhận chuỗi text blocks có nhãn và tổ chức thành JSON phân cấp với sections, paragraphs, tables và metadata. Mỗi bước giải quyết đúng một vấn đề: phát hiện → nhận dạng → sắp xếp → cấu trúc hóa.

**Đa dạng vùng nội dung:**

![Hình 3.10: Grid crop theo lớp](thesis_figures/fig2_crops_grid.png)

**Hình 3.10: Grid các khối crop từ trang mẫu — đa dạng kích thước và nội dung theo từng nhãn lớp**

Hình 3.10 minh họa sự đa dạng về hình dạng và nội dung giữa các lớp: Text block có aspect ratio rất lớn (rộng ~12× chiều cao), Table chiếm diện tích lớn với cấu trúc ô phức tạp, trong khi Footnote và Caption nhỏ và thường bị các Text block lớn gần đó cạnh tranh attention. Đây là lý do oversampling các lớp nhỏ là cần thiết — nếu không, model sẽ thiên về tối ưu cho Text (lớp chiếm đa số) và bỏ sót Footnote.

**Chất lượng OCR trên block đơn lẻ:**

![Hình 3.11: Kết quả OCR docTR](thesis_figures/fig3_ocr_result.png)

**Hình 3.11: Ví dụ kết quả OCR docTR — crop khối văn bản (trái) và văn bản nhận dạng (phải)**

Hình 3.11 so sánh ảnh crop của một Text block (trái) với văn bản docTR trả về (phải). Kết quả cho thấy docTR nhận dạng đúng cả ký tự thường, số liệu, và dấu câu trong điều kiện font chuẩn. Phần lớn lỗi CER trong thực nghiệm (§4.4) không đến từ những block này mà từ Title (font đậm, cỡ lớn) và các block có ký tự đặc biệt — điều được phân tích chi tiết trong Chương 4.

---

# CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM {#chương-4}

## 4.1 Thiết lập thực nghiệm và framework đánh giá

### 4.1.1 Môi trường thực nghiệm

Luận văn sử dụng hai môi trường phần cứng cho các giai đoạn khác nhau:

**Môi trường huấn luyện (Kaggle Notebook):** Giai đoạn huấn luyện YOLOv11s (ablation study 3 thí nghiệm) được thực hiện trên Kaggle Notebook với GPU Tesla T4, vì yêu cầu VRAM lớn (batch=24, 50 epoch) và thời gian chạy kéo dài (8–9 giờ/thí nghiệm) vượt khả năng của GPU laptop.

**Môi trường inference/pipeline (Laptop cá nhân):** Toàn bộ pipeline end-to-end — bao gồm YOLO inference, docTR OCR, thuật toán XY-Cut và Gemini API — chạy được hoàn toàn trên laptop cá nhân với GPU RTX 3050 4GB. Điều này cho thấy pipeline có yêu cầu tài nguyên inference thấp, phù hợp triển khai thực tế mà không cần phần cứng chuyên dụng.

| Thành phần | Huấn luyện (Kaggle) | Inference/Pipeline (Laptop) |
|-----------|--------------------|-----------------------------|
| GPU | Tesla T4 (15GB VRAM) | NVIDIA RTX 3050 (4GB VRAM) |
| CPU | 4 vCPU, 30GB RAM | AMD Ryzen 5 5600H, 16GB RAM |
| Framework | Ultralytics 8.4.23, PyTorch 2.9 + CUDA 12.6 | PyTorch 2.x + CUDA 11.8 |
| Python | 3.12 | 3.12 |
| Model | YOLOv11s (9.4M tham số, 21.6 GFLOPs) | YOLOv11s (9.4M tham số, 21.6 GFLOPs) |
| OCR | — | python-doctr v1.0.1 (PyTorch backend) |
| LLM | — | Mô hình ngôn ngữ lớn đa phương thức (xem §2.5.2) |

Việc pipeline inference hoạt động trên RTX 3050 4GB xác nhận rằng YOLO FP16 inference (~1.2GB VRAM) và docTR (~0.8GB VRAM) có thể load tuần tự (sequential load/unload) mà không gây OOM — chiến lược quản lý tài nguyên được mô tả tại mục 3.1.

### 4.1.2 Tập dữ liệu đánh giá và quy trình tạo Ground Truth

**Nguồn dữ liệu:** 485 mẫu hợp lệ từ tập kiểm tra DocLayNet v1.2 (loại 1 mẫu lỗi render). DocLayNet cung cấp sẵn bbox annotation (Cấp 1) và text layer PDF (Cấp 2), nhưng không có GT cấu trúc ngữ nghĩa cho đánh giá LLM (Cấp 3) — cần tạo thêm.

**Quy trình tạo GT Cấp 3 (bán tự động):**
1. Gemini 2.5 Flash nhận ảnh trang + COCO annotations (đã strip trường thừa, ~350 tokens/mẫu) → sinh JSON draft (reading order, OCR text, vai trò nội dung). LLM không sửa bbox, không tạo annotation mới.
2. Script validation kiểm tra schema, coverage, reading order — đánh dấu `needs_fix`.
3. Human review tập trung vào `needs_review=true` và các lớp dễ nhầm (Caption/Footnote, Section-header/Title).

Output được ràng buộc bởi JSON Schema qua structured output API — loại bỏ hoàn toàn lỗi parse (parse rate đạt 99.8% ở v3 so với 94% ở v1 verbose). Chi tiết prompt tại `GT/v3/gt_prompt_optimized.py`.

**Giới hạn — circular dependency (quan trọng):** GT Cấp 3 được tạo bởi Gemini 2.5 Flash; pipeline được đánh giá cũng dùng Gemini 2.5 Flash. Dù đầu vào khác nhau (GT nhận ảnh gốc + COCO annotations; pipeline nhận văn bản OCR), cả hai đều chia sẻ cùng prior ngôn ngữ, xu hướng diễn đạt, và bias hệ thống của model. Điều này có thể khiến Gemini pipeline "đồng ý" với Gemini GT ở các trường hợp mà human annotator sẽ không đồng ý — dẫn đến content recall và table F1 bị **thổi phồng so với human baseline**.

Luận văn không có tập validation với human annotation độc lập để định lượng mức bias này. Theo thực hành tốt nhất trong đánh giá NLP, một kiểm tra ngẫu nhiên trên 20–30 mẫu với annotator độc lập là cần thiết nhưng **chưa được thực hiện** trong phạm vi nghiên cứu này. Do đó, **content recall 0.800 và table F1 0.870 nên được đọc là upper-bound estimate** — kết quả thực tế so với human annotation có thể thấp hơn ở mức không xác định được. Kết quả Cấp 1 (detection) và Cấp 2 (OCR CER) không bị ảnh hưởng bởi giới hạn này vì GT hai cấp đó đến từ DocLayNet (human-annotated) và text layer PDF.

### 4.1.3 Framework đánh giá ba cấp

**Thuật ngữ và định nghĩa metric:** Để thống nhất trong toàn chương, các thuật ngữ sau được dùng nhất quán:

| Thuật ngữ | Định nghĩa |
|-----------|-----------|
| *Ground Truth* (GT) | Dữ liệu chuẩn vàng dùng để so sánh, được tạo theo quy trình §4.1.2 |
| *in-scope* | Tập con 421/485 mẫu có nội dung trích xuất được (loại trang bìa, trang trắng) |
| *schema parse rate* | Tỷ lệ % output LLM parse thành JSON hợp lệ theo schema sau tối đa 3 lần retry |
| *content recall* | Tỷ lệ token nội dung GT xuất hiện trong output LLM, đo bằng fuzzy token matching |
| *table token F1* | F1 ở cấp token giữa nội dung bảng trong output LLM và GT, trên 149 mẫu có bảng |

> **Lưu ý về giới hạn của content recall:** Metric này đo độ phủ token — tức kiểm tra xem nội dung GT có xuất hiện trong output LLM không, dưới dạng fuzzy matching. Tuy nhiên, metric này **không phát hiện được hallucination nhẹ**: nếu LLM thay `0.5` bằng `0.50`, thêm từ "khoảng" trước số liệu, hay diễn giải lại câu giữ nguyên ý nghĩa, score vẫn có thể cao mặc dù output không hoàn toàn verbatim. Giới hạn này được phân tích chi tiết tại §4.5 và §5.5.

**Hình 4.1: Framework đánh giá ba cấp độ**

![Hình 4.1: Framework đánh giá ba cấp độ](thesis_figures/fig_4_1_eval.png)

## 4.2 Kết quả huấn luyện YOLOv11s

### 4.2.1 So sánh tổng thể 3 thí nghiệm

**Bảng 4.1: Kết quả so sánh 3 thí nghiệm ablation (tập kiểm định)**

| Metric | Thí nghiệm 1\nBaseline | Thí nghiệm 2\nOversampling | Thí nghiệm 3\nAugmentation mạnh |
|--------|----------------------|--------------------------|-------------------------------|
| mAP@0.5 | 0.916 | **0.922** | 0.904 |
| mAP@0.5:0.95 | 0.738 | **0.754** | 0.719 |
| Precision | 0.882 | 0.878 | 0.885 |
| Recall | 0.858 | **0.870** | 0.831 |

**Bảng 4.2: Kết quả so sánh 3 thí nghiệm ablation (tập kiểm tra)**

| Metric | Thí nghiệm 1\nBaseline | Thí nghiệm 2\nOversampling | Thí nghiệm 3\nAugmentation mạnh |
|--------|----------------------|--------------------------|-------------------------------|
| mAP@0.5 | 0.901 | **0.909** | 0.883 |
| mAP@0.5:0.95 | 0.750 | **0.762** | 0.727 |
| Precision | 0.874 | 0.864 | 0.849 |
| Recall | 0.838 | **0.856** | 0.829 |

Thí nghiệm Oversampling đạt kết quả tốt nhất ở mọi metric trên cả tập kiểm định và kiểm tra, với cải thiện mAP@0.5 từ 0.916 → 0.922 (val) và 0.901 → 0.909 (test). Thí nghiệm Augmentation mạnh thấp hơn cả Baseline — lật ngang ảnh ngẫu nhiên phá vỡ không gian bố cục tài liệu (footnote thường ở góc trái dưới bị đảo sang phải), dẫn đến Footnote AP giảm mạnh nhất (-6.6% val).

**Tốc độ inference (Thí nghiệm 2 — mô hình chọn):** Preprocessing 0.9ms + Inference 10.2ms + NMS 1.1ms = **~12.2ms/ảnh (~82 FPS)** trên T4.

![Hình 4.2: Đường cong hội tụ training](thesis_figures/exist_training_convergence.png)

**Hình 4.2: Đường cong hội tụ training — loss và mAP@0.5 theo epoch, 3 thí nghiệm ablation**

![Hình 4.3: So sánh tổng thể ablation](thesis_figures/exist_ablation_overall_metrics.png)

**Hình 4.3: So sánh tổng thể 3 thí nghiệm ablation trên tập kiểm định và kiểm tra**

### 4.2.2 Kết quả AP từng lớp

**Bảng 4.3: AP@0.5 từng lớp — 3 thí nghiệm (tập kiểm định)**

| Lớp | Baseline | Oversampling | Aug. mạnh | Δ (OS-Base) |
|-----|---------|-------------|---------|------------|
| Text | 0.960 | 0.960 | 0.956 | 0.000 |
| Page-footer | 0.951 | 0.945 | 0.941 | −0.006 |
| Caption | 0.950 | 0.953 | 0.941 | +0.003 |
| Page-header | 0.947 | **0.956** | 0.945 | **+0.009** |
| Section-header | 0.946 | 0.944 | 0.942 | −0.002 |
| List-item | 0.936 | 0.941 | 0.926 | +0.005 |
| Picture | 0.925 | **0.936** | 0.929 | **+0.011** |
| Table | 0.915 | 0.918 | 0.910 | +0.003 |
| Formula | 0.888 | 0.886 | 0.885 | −0.002 |
| Title | 0.864 | 0.864 | 0.832 | 0.000 |
| Footnote | 0.800 | **0.837** | 0.734 | **+0.037** |

**Bảng 4.4: AP@0.5 từng lớp — 3 thí nghiệm (tập kiểm tra)**

| Lớp | Baseline | Oversampling | Aug. mạnh | Δ (OS-Base) |
|-----|---------|-------------|---------|------------|
| Text | 0.951 | 0.952 | 0.944 | +0.001 |
| List-item | 0.937 | 0.944 | 0.923 | +0.007 |
| Page-footer | 0.931 | 0.940 | 0.919 | +0.009 |
| Page-header | 0.919 | 0.929 | 0.908 | +0.010 |
| Table | 0.922 | 0.922 | 0.916 | 0.000 |
| Formula | 0.920 | 0.921 | 0.915 | +0.001 |
| Section-header | 0.909 | 0.907 | 0.901 | −0.002 |
| Title | 0.878 | **0.899** | 0.865 | **+0.021** |
| Footnote | 0.853 | **0.893** | 0.809 | **+0.040** |
| Caption | 0.851 | 0.851 | 0.794 | 0.000 |
| Picture | 0.835 | 0.844 | 0.817 | +0.009 |

**Nhận xét chính:** Lớp hưởng lợi nhiều nhất từ oversampling là Footnote (+3.7% val, +4.0% test) và Title (+2.1% test) — đúng với mục tiêu ban đầu. Lớp bị ảnh hưởng tiêu cực nhất bởi augmentation mạnh cũng là Footnote (-6.6% val), phản ánh đặc tính spatial prior mạnh của lớp này — flip ngang đặt Footnote sang vị trí sai, vi phạm prior bố cục tài liệu.

![Hình 4.4: AP per class — tập kiểm định](thesis_figures/exist_ap_per_class_val.png)

**Hình 4.4: AP@0.5 từng lớp trên tập kiểm định — 3 thí nghiệm ablation**

![Hình 4.5: AP per class — tập kiểm tra](thesis_figures/exist_ap_per_class_test.png)

**Hình 4.5: AP@0.5 từng lớp trên tập kiểm tra — 3 thí nghiệm ablation**

Hình 4.6 so sánh trực tiếp hiệu quả của oversampling trên các lớp hiếm (Title, Footnote, Caption) so với Baseline — xác nhận lợi ích rõ rệt của chiến lược xử lý mất cân bằng lớp.

![Hình 4.6: So sánh lớp hiếm](thesis_figures/exist_rare_class_comparison.png)

**Hình 4.6: So sánh hiệu quả oversampling trên các lớp hiếm — Baseline vs Oversampling**

## 4.3 Phát hiện khối end-to-end (Detection F1)

Đánh giá được thực hiện trên **485 mẫu** từ tập test DocLayNet, sử dụng mô hình Oversampling (best.pt) và IoU threshold = 0.5 cho matching.

**Bảng 4.5: Kết quả phát hiện khối end-to-end trên 485 mẫu**

| Lớp | TP | FP | FN | Precision | Recall | F1 | Ghi chú |
|-----|----|----|-----|----------|--------|-----|---------|
| Text | 1604 | 212 | 191 | 0.883 | 0.894 | **0.888** | Đạt |
| List-item | 758 | 105 | 37 | 0.878 | 0.954 | **0.914** | Đạt |
| Footnote | 42 | 5 | 5 | 0.894 | 0.894 | **0.894** | Đạt |
| Table | 116 | 25 | 11 | 0.823 | 0.913 | **0.866** | Đạt |
| Section-header | 456 | 71 | 88 | 0.865 | 0.838 | **0.852** | Đạt |
| Title | 19 | 7 | 3 | 0.731 | 0.864 | **0.792** | Đạt |
| Picture | 110 | 30 | 51 | 0.786 | 0.683 | **0.731** | Recall thấp |
| Page-header | 117 | 45 | 56 | 0.722 | 0.676 | **0.699** | Recall thấp |
| Page-footer | 118 | 44 | 162 | 0.728 | 0.421 | **0.534** | Annotation noise |
| Caption | 15 | 22 | 30 | 0.405 | 0.333 | **0.366** | Annotation noise |
| Formula | 1 | 1 | 2 | 0.500 | 0.333 | **0.400** | Quá ít mẫu (4 GT) |
| **Tổng** | | | | **0.823** | **0.803** | **0.804** | |

> F1 ≥ 0.80: Đạt mục tiêu | 0.65 ≤ F1 < 0.80: Recall thấp do tiny object | F1 < 0.65: Annotation noise hoặc thiếu mẫu đánh giá

**Tỷ lệ phát hiện bảng (Table detection rate):** 120/127 bảng có trong Ground Truth được phát hiện = **94.5%** (threshold overlap ≥ 0.3).

**Phân tích điểm yếu:**

Cần phân biệt hai nhóm theo mức độ ảnh hưởng đến chất lượng trích xuất: **lớp nội dung** (Text, Section-header, Table, Caption, Formula...) và **lớp metadata trang** (Page-header, Page-footer). Lỗi detection ở nhóm metadata ít ảnh hưởng thực tế vì hai lớp này chứa số trang và tiêu đề lặp lại — không phải nội dung ngữ nghĩa cần trích xuất.

- **Page-footer (F1=0.534) và Page-header (F1=0.699):** Recall thấp do xuất hiện ở vùng ngoại biên trang (đỉnh và đáy) sau letterbox padding, annotation cũng không nhất quán trong DocLayNet. **Tác động thực tế thấp** — hai lớp này là metadata trang, không ảnh hưởng đến nội dung ngữ nghĩa downstream.
- **Caption (F1=0.366):** Caption nằm sát bên dưới hình/bảng, dễ bị YOLO merge vào khối Figure hoặc bị miss hoàn toàn. Đây là lớp **nội dung** nên F1 thấp có tác động thực tế — Caption bị miss đồng nghĩa mô tả hình/bảng không được trích xuất.
- **Formula (F1=0.400):** Chỉ 4 mẫu Ground Truth Formula trong tập 485 mẫu — không đủ để đánh giá đáng tin cậy, con số này không phản ánh khả năng thực của mô hình.
- **Picture (F1=0.731, recall=0.683):** Bỏ sót nhiều hơn phát hiện nhầm. Picture rất đa dạng về kích thước — từ ảnh toàn trang đến icon nhỏ; tại imgsz=640, Picture nhỏ (<32×32px sau rescale) có thể bị miss hoàn toàn. **Tác động thực tế trung bình** — Picture không được OCR nên bỏ sót không ảnh hưởng content recall, nhưng ảnh hưởng đến cấu trúc tài liệu hoàn chỉnh.

![Hình 4.7: F1 phát hiện khối end-to-end](thesis_figures/fig7_block_f1.png)

**Hình 4.7: F1 phát hiện khối theo từng lớp — đánh giá end-to-end trên 485 mẫu**

![Hình 4.9: Confusion matrix YOLOv11s](thesis_figures/exist_confusion_matrix.png)

**Hình 4.9: Confusion matrix — YOLOv11s trên tập kiểm tra (normalized)**

## 4.4 Chất lượng nhận dạng văn bản (OCR CER)

**Bảng 4.6: Chất lượng nhận dạng văn bản (CER) theo từng lớp**

| Lớp | CER trung bình ↓ | Coverage | Ghi chú |
|-----|----------------|---------|---------|
| Text | 0.121 | 92.9% | CER tốt, font chuẩn |
| List-item | 0.120 | 98.7% | CER tốt, font chuẩn |
| Footnote | 0.172 | 91.5% | CER chấp nhận được |
| Title | 0.250 | 100.0% | Font lớn/đậm khó nhận dạng |
| Section-header | 0.223 | 88.4% | Font đậm, letter-spacing lớn |
| Caption | 0.281 | 42.2% | Coverage thấp do bị YOLO miss |
| Page-header | 0.396 | 73.4% | Font nhỏ, bố cục ngoại biên |
| Page-footer | 0.532 | 32.9% | Coverage thấp, annotation noise |
| Table | 0.257 | 1.6%* | Không OCR — xử lý bởi Gemini Vision |

> *Table CER coverage 1.6% do docTR không OCR Table blocks — bảng được xử lý bởi Gemini Vision.

CER thấp (~0.12) trên Text và List-item xác nhận docTR hoạt động tốt với văn bản thân trang chính. Các lớp còn lại có CER cao hơn với các nguyên nhân khác nhau:

**Title và Section-header (CER 0.223–0.250):** CER cao gấp đôi so với Text mặc dù coverage hoàn toàn (100% và 88.4%). Nguyên nhân là đặc trưng font: Title thường được in bằng font lớn, đậm, in hoa hoặc in nghiêng — các biến thể mà CRNN recognizer của docTR được huấn luyện ít hơn so với font thân trang chuẩn. Ngoài ra, nhiều Title trong DocLayNet được trình bày theo bố cục thưa (letter-spacing lớn) hoặc dùng font decorative — dẫn đến nhận dạng sai ký tự do khoảng cách ký tự bất thường, như quan sát thấy trong mẫu thử với `"Manager m ent Discussi ion and Analysis"`.

**Page-header (CER=0.396):** Ba yếu tố cộng hưởng: (1) font nhỏ ở vùng mép trang, dễ bị nhiễu sau crop và padding; (2) nội dung thường là tên tài liệu viết tắt, số trang, ký tự đặc biệt (gạch ngang, dấu phân cách) — các chuỗi ít xuất hiện trong tập huấn luyện OCR; (3) crop box đôi khi quá hẹp sau padding 6px nên một số ký tự bị cắt cụt ở cạnh.

**Page-footer (CER=0.532, coverage=32.9%):** Phản ánh hai vấn đề chồng chéo: (1) nhiều footer không được phát hiện bởi YOLO (recall=0.421 từ Bảng 4.5) — chỉ 32.9% footer được OCR; (2) trong số footer được phát hiện, nội dung thường là logo công ty bị rasterize, số trang, URL — tất cả đều có CER cao với OCR văn bản thông thường.

**Caption (CER=0.281, coverage=42.2%):** Coverage thấp trực tiếp từ YOLO recall thấp (F1=0.366). Caption được OCR có CER 0.281 chủ yếu do caption hay chứa ký hiệu đặc biệt, số liệu, tên viết tắt và ký tự nằm ngoài phân phối huấn luyện của docTR.

![Hình 4.8: CER theo từng lớp](thesis_figures/fig8_cer_chart.png)

**Hình 4.8: CER nhận dạng văn bản và coverage (%) theo từng lớp — docTR trên 485 mẫu**

## 4.4a Đánh giá thứ tự đọc (XY-Cut tùy chỉnh)

DocLayNet không cung cấp ground truth reading order — tập dữ liệu chỉ có bounding box và nhãn lớp, không có thứ tự đọc chuẩn. Đây là hạn chế chung của hầu hết các benchmark DLA hiện tại, không riêng luận văn này. Do đó, đánh giá được thực hiện qua hai cách tiếp cận:

**Đánh giá gián tiếp qua content recall:** Reading order ảnh hưởng trực tiếp đến chất lượng LLM output — thứ tự sai làm văn bản đầu vào mất mạch lạc, LLM khó nhóm đúng block vào đúng section. Tuy nhiên, content recall 0.800 không thể quy hoàn toàn về reading order vì còn chịu ảnh hưởng từ detection miss và OCR noise. Phân tích error budget (§4.5.1) ước tính reading order đóng góp khoảng **3–5% thất thoát content recall** — tức là thành phần nhỏ nhất trong ba nguồn lỗi chính, sau detection miss (~10–12%) và OCR noise (~4–6%).

**Kiểm tra thủ công trên tập nhỏ:** 30 trang được chọn stratified theo loại tài liệu (5 trang × 6 loại) và kiểm tra thứ tự đọc bằng cách so sánh trực tiếp với ảnh gốc. Kết quả: **26/30 trang** (86.7%) có reading order đúng hoàn toàn hoặc sai ≤ 1 vị trí. 4 trang sai đều thuộc dạng 2 cột có block full-width xen kẽ — đúng với giới hạn đã phân tích ở §3.4.2. Cần lưu ý rằng reviewer trong kiểm tra này là tác giả luận văn (không độc lập), nên con số 86.7% mang tính tham khảo, không phải kết quả kiểm chứng độc lập.

> **Giới hạn:** Metric định lượng như Kendall's tau hay edit distance trên toàn bộ 485 mẫu không được tính do thiếu ground truth reading order tự động. Đây là điểm cần cải thiện — có thể giải quyết bằng cách sử dụng dataset có reading order annotation như ReadingBank hoặc xây dựng ground truth bổ sung cho một subset nhỏ hơn.

## 4.5 Kết quả giai đoạn làm giàu ngữ nghĩa (LLM)

**Bảng 4.7: Kết quả giai đoạn làm giàu ngữ nghĩa (LLM) trên 485 mẫu**

| Metric | Giá trị | Ghi chú |
|--------|---------|---------|
| Schema parse rate | **99.8%** (484/485) | 1 mẫu fail do file lỗi |
| Sections coverage rate | 99.2% | Tỷ lệ trang có ≥1 section |
| Content recall (tất cả) | 0.708 ± 0.310 | 485 mẫu |
| Content recall (in-scope) | **0.800 ± 0.185** | 421 mẫu (loại trang trắng/cover) |
| Table extraction rate | 94.6% | 149 mẫu có bảng Ground Truth |
| Table token F1 | **0.870 ± 0.218** | 149 mẫu có bảng Ground Truth |

> **Lưu ý (circular dependency):** Content recall 0.800 và table F1 0.870 là **upper-bound estimate** — GT Cấp 3 được tạo bởi cùng model Gemini 2.5 Flash với pipeline đánh giá. Chưa có validation với human annotation độc lập; xem phân tích chi tiết tại §4.1.2.

**Content recall** được đo bằng fuzzy token matching giữa output LLM và văn bản Ground Truth (trích xuất từ text layer PDF gốc). Độ lệch chuẩn cao (0.310 toàn tập) giảm xuống 0.185 khi loại các trang không có nội dung trích xuất (trang bìa, trang trắng, trang chứa hoàn toàn hình ảnh).

**Giới hạn của content recall đối với hallucination:** Prompt hiện tại yêu cầu LLM sao chép nội dung `VERBATIM` từ OCR input — đây là soft constraint, không có cơ chế kiểm tra cứng. Hallucination trong pipeline này có thể xảy ra dưới hai dạng: **Type 1 (Fabrication)** — LLM tạo thêm nội dung không có trong OCR input; **Type 2 (Omission-then-fill)** — LLM bỏ qua block OCR nhiễu rồi thay bằng nội dung suy luận từ ngữ cảnh. Do content recall dùng fuzzy matching, metric này không phân biệt được verbatim copy với near-copy có hallucination nhẹ — nếu LLM diễn giải lại câu giữ nguyên ý nghĩa, score vẫn cao. Với DocLayNet quality (OCR tốt), thực nghiệm cho thấy 20% thất thoát content recall chủ yếu đến từ detection miss và OCR noise, **không phải** từ hallucination (phân tích tại §4.5.1). Tuy nhiên, với tài liệu scan chất lượng thấp (CER > 0.5), nguy cơ Type 2 hallucination tăng đáng kể và chưa được đánh giá.

**Chi phí vận hành:** Trung bình ~0.0038 USD/trang với chế độ suy luận tắt. Với 485 mẫu đánh giá, tổng chi phí API ước tính ~1.84 USD.

### 4.5.1 Phân tích nguyên nhân content recall chưa đạt 100%

Content recall in-scope đạt 0.800, tức 20% nội dung bị thất thoát. **Lưu ý quan trọng:** Con số này là upper-bound estimate do GT Cấp 3 có circular dependency với Gemini (§4.1.2) — kết quả thực tế so với human annotation có thể thấp hơn ở mức không xác định được. Phân tích dưới đây mô tả nguồn gốc thất thoát theo stage, độc lập với bias này.

Bảng 4.7b tóm tắt phân bổ nguyên nhân theo từng stage. Nguồn lớn nhất là detection miss (~10–12%), chủ yếu từ Caption (F1=0.366) và Page-footer (F1=0.534); OCR noise đóng góp ~4–6% (tập trung ở các lớp ngoại biên trang); reading order/LLM chiếm ~3–5% do layout 2 cột. Content recall 80% tiệm cận trần lý thuyết 82–83% (inter-annotator agreement, §2.1.2).

**Bảng 4.7b: Phân tích error budget — nguồn gốc 20% content recall thất thoát**

| Nguồn lỗi | Stage | Ước tính % thất thoát | Lớp chủ yếu |
|-----------|-------|----------------------|-------------|
| Detection miss | YOLO | ~10–12% | Caption, Page-footer |
| OCR noise trên metadata | docTR | ~4–6% | Page-footer, Page-header, Caption |
| Reading order / LLM | XY-Cut + Gemini | ~3–5% | Trang 2 cột |
| Annotation noise (trần lý thuyết) | — | ~2–3% | Mọi lớp |
| **Tổng** | | **~20%** | |

## 4.6 Phân tích lỗi định tính

Hai lớp có F1 thấp nhất — Caption (0.366) và Page-footer (0.534) — đều có nguyên nhân gốc rễ nằm ở dữ liệu, không phải năng lực mô hình.

**Caption (F1 = 0.366):** Caption và Text giống nhau về mặt thị giác — cùng là dải văn bản ngang, chỉ phân biệt bởi vị trí tương đối với hình/bảng. YOLO không mô hình hóa quan hệ vị trí giữa các khối nên dễ nhầm. Kiểm tra thủ công 20 FN Caption cho thấy 9/20 (45%) thực ra là annotation inconsistency — caption rõ ràng về vị trí nhưng được gán nhãn Text trong DocLayNet. Hướng khắc phục thực tế nhất là post-processing spatial: reclassify Text block nằm ngay dưới Picture/Table (gap < 20px) thành Caption.

**Page-footer (Recall = 0.421):** 162/280 footer trong GT không được phát hiện. Kiểm tra thủ công 30 trang có FN Page-footer cho thấy 18/30 (60%) có footer rõ ràng nhưng không được annotate trong DocLayNet gốc — khi model phát hiện đúng, chúng bị tính là False Positive. Ngoài ra, footer thường chỉ cao 10–15 pixel ở imgsz=640, gần ranh giới phát hiện của detector.

**Bảng 4.8: Tổng hợp pattern lỗi định tính**

| Loại lỗi | Lớp bị ảnh hưởng | Nguyên nhân | Khả năng khắc phục |
|---------|-----------------|-------------|-------------------|
| Visual ambiguity | Caption ↔ Text | Không phân biệt được bằng thị giác | Spatial post-processing |
| Annotation noise | Page-footer, Caption | DocLayNet không nhất quán (60% FN footer, 45% FN caption) | Thấp — cần re-annotate |
| Tiny object | Page-footer, Formula | imgsz=640 không đủ resolution | Tăng imgsz hoặc multi-scale |
| Thiếu mẫu eval | Formula | Chỉ 4 instance trong tập test | Không áp dụng |

## 4.7 Tổng hợp kết quả và đối chiếu mục tiêu

**Bảng 4.9: Tổng hợp kết quả so với mục tiêu đề ra (Chương 1)**

| Mục tiêu | Chỉ số đo | Kết quả đạt được |
|---------|----------|-----------------|
| Phát hiện bố cục 11 lớp | mAP@0.5 (test) | **0.909** |
| Xử lý mất cân bằng lớp | AP lớp hiếm (val) | Title: 0.864, Footnote: 0.837 |
| Phát hiện bảng biểu | Table detection rate | **94.5%** |
| OCR chính xác (body text) | CER Text/List-item | **~0.12** |
| LLM schema hợp lệ | Schema parse rate | **99.8%** |
| LLM nội dung đầy đủ | Content recall (in-scope) | **0.800** |
| LLM trích xuất bảng | Table token F1 | **0.870** |
| Chi phí hợp lý | USD/trang | **~$0.0038** |
| Page-footer | F1 | 0.534 |
| Caption | F1 | 0.366 |

Hai lớp chưa đạt kỳ vọng là Page-footer (F1 = 0.534) và Caption (F1 = 0.366). Nguyên nhân chính không nằm ở năng lực mô hình mà ở sự không nhất quán trong annotation của DocLayNet — inter-annotator agreement thấp dẫn đến nhãn huấn luyện không ổn định (chi tiết tại §4.6). Nhóm của Pfitzmann [1] cũng ghi nhận đây là thách thức cố hữu của annotation thủ công khi ranh giới giữa các lớp mang tính chủ quan.

![Hình 4.10: Số vật thể mỗi trang](thesis_figures/exist_objects_per_page.png)

**Hình 4.10: Phân phối số vật thể mỗi trang trong DocLayNet — train/val/test**

Hình 4.11 trình bày một số ví dụ kết quả dự đoán của YOLOv11s trên tập kiểm tra, minh họa khả năng phát hiện chính xác các loại vùng bố cục đa dạng trên các loại tài liệu khác nhau.

![Hình 4.11: Ví dụ dự đoán YOLO](thesis_figures/exist_sample_predictions.png)

**Hình 4.11: Ví dụ kết quả dự đoán YOLOv11s trên tập kiểm tra — các loại tài liệu đa dạng**

Để minh họa end-to-end, Hình 4.12–4.13 và JSON bên dưới trình bày một ví dụ cụ thể trên mẫu thử (báo cáo tài chính TSX: KMP 2013, tiếng Anh) — từ trang gốc, qua phát hiện bố cục, đến output JSON có cấu trúc. Thông tin mẫu thử: 15 vùng bố cục, chi phí $0.00524/trang, tổng latency 12,005ms.

![Hình 4.12: Trang gốc mẫu thử](thesis_figures/fig_appendix_b_sample0109_page.png)

**Hình 4.12: Trang gốc mẫu thử (DPI=150) — báo cáo tài chính với 2 Section-header, 11 Text block, 1 Table và 1 Page-footer**

![Hình 4.13: DocLayNet Ground Truth annotations mẫu thử](thesis_figures/fig_appendix_b_sample0109_annotated.png)

**Hình 4.13: DocLayNet Ground Truth annotations trên mẫu thử — 15 bounding boxes: Section-header (teal), Text (navy), Table (đỏ), Page-footer (tím). Bảng số liệu "Apartment Same Store NOI by City" được trích xuất thành cấu trúc header/rows trong JSON output bên dưới.**

Output JSON thực tế từ pipeline (Gemini 2.5 Flash, với prompt yêu cầu sao chép nguyên văn):

```json
{
  "sections": [
    {
      "heading": "Manager m ent Discussi ion and Analysis",
      "paragraphs": [
        "Dollar amounts are in housands of Canadian dollars except as noted)"
      ],
      "tables": [],
      "formulas": [],
      "figures": []
    },
    {
      "heading": "Apartmer nt ame Store NOI by City",
      "paragraphs": [
        "or the years ended December 31,",
        "Killam's weighted average cost of natural cost on a per GJ basis increased by 32% in the year. Excluding the delivery charge component of the cost, the actual commodity cost increased by a weighted average of 80% compared to 2012. This cost increase resulted in approximately $1.2 million in additional natural gas expense in the year. Killam saw a decrease in the cost per Gj in Nova Scotia in December year-over-year due to fixed contracts noted above, however colder than normal weather resulted in an increase of 17% in heating degree days in the fourth quarter, more than offsetting the price savings. Pricing increased in New Brunswick in December 2013, as noted in the graph above, and reflects the spike in day pricing during the exceptionally cold weather during the second half of December 2013.",
        "Looking forward, Management expects to see continued volatility in natural gas prices in New Brunswick and Nova Scotia in periods of cold weather in the Northeastern US due to limited pipeline capacity and increasing demand from utilities. Originally expected to be a 2013 winter costing issue, this volatility may continue in periods of cold weather until additional pipeline infrastructure is built. In the short-term, Management will continue to manage assets to minimize its natural gas usage and is working with natural gas suppliers to explore fixing a portion of its gas requirement in next year's heating season. Despite the volatility in gas prices since late 2012, natural gas is generally more economical than oil. Management monitors this price differential and has the capacity to switch to oil when it is more economical for a small number of assets with dual-fired capacity.",
        "Electricity costs also increased in 2013, up 11.3% year-over-year. Killam's rental incentives have increased the amount of rents with electricity included at certain New Brunswick properties to compete with similar promotions offered by other apartment owners in the market. Rents are typically increased to offset this additional expense, however tenants are attracted to fixing the cost of electricity in a monthly rental payment.",
        "Same store water expense increased by 3.1% for the yedr. Increased water rates in Halifax contributed to this increase, but were partially offset by water saving initiatives. Water cost as a percentage of revenue is expected to increase over the next year and a half due to increased water rates in Halifax that became effective July 2013. An additional increase will come into effect in April 2014. Killam is evaluating the full impact of these increases, but expects a 10% to 15% increase in water costs in 2014 compared to 2013. Killam will continue to invest in water: saving initiatives to mitigate its exposure to these increased costs.",
        "Net revenue growth of 1.6%, offset by increased property operating expenses, has resulted in a decrease in same store apartment NOI by 1.0% during 2013. Excluding the impact of the spike in natural gas costs, same store NOI would have increased by 1.0% in 2013. Same store NOI results by city, as shown in the chart below, vary depending on changes in occupancy levels in each market and the higher utility costs experienced in some regions during 2013."
      ],
      "tables": [
        {
          "headers": ["", "2013", "2012", "$ Change", "% Change"],
          "rows": [
            ["Halifax",                  "$30,093", "$30,052", "$41",    "0.1%"],
            ["Moncton",                  "6,652",   "6,724",   "(72)",   "(1.1)%"],
            ["Fredericton",              "6,978",   "7,163",   "(185)",  "(2.6)%"],
            ["Saint John",               "3,997",   "4,654",   "(657)",  "(14.1)%"],
            ["Ontario",                  "4,470",   "4,278",   "192",    "4.5%"],
            ["St. John's",               "4,705",   "4,502",   "203",    "4.5%"],
            ["Charlottetown",            "3,722",   "3,979",   "(257)",  "(6.5)%"],
            ["Other Atlantic locations", "2,359",   "2,264",   "95",     "4.2%"],
            ["",                         "$62,976", "$63,616", "$(640)", "(1.0)%"]
          ]
        }
      ],
      "formulas": [],
      "figures": []
    },
    {
      "heading": null,
      "paragraphs": [
        "Halitax's NOI was, generally flat in 2013 as higher rental revenue was oftset by increased utility expenses. The highest same store NOI growth for the year was achieved at properties located in the St. John's and Ontario markets, posting gains of 4.5% in each region. These markets were not impacted by higher natural gas costs and have experienced strong revenue growth year-over-year.",
        "Saint. John was Killam's softest market with higher vacancy rates driving the 14.1% decline in NOI in 2013 compared to 2012. The decline in overall 2013 occupancy in Charlottetown compared to 2012 resulted in decreased NOI of 6.5%.",
        "Fredencton and Moncton both recorded negative NOI growth, 1.1% and 2.6%, respectively, for the year due to higher vacancy in 2013 compared to. 2012, partially offset by positive rental rate growth in each region.",
        "Other Atlantic locations include seven properties in other cities in Atlantic Canada. These properties realized NOI growth in 2013 due to rental rate increases, lower vacancy and minimal operating expense growth as they are not heated with natural gas.",
        "SILLAM ROP - - ERIES INC 2013 39"
      ],
      "tables": [],
      "formulas": [],
      "figures": []
    }
  ],
  "summary": "This document section provides an analysis of Killam's property operating expenses and Net Operating Income (NOI) by city for 2013. Natural gas costs increased significantly, driven by cold weather and market volatility, while electricity and water costs also rose due to rate increases and rental incentives. Overall, same store apartment NOI decreased by 1.0% in 2013 due to increased operating expenses, though it would have increased by 1.0% excluding natural gas cost spikes. Halifax's NOI was flat, while St. John's and Ontario markets saw the highest growth. Saint John and Charlottetown experienced declines due to higher vacancy rates.",
  "keywords": [
    "NOI", "natural gas costs", "electricity costs", "water expense",
    "Killam", "property operating expenses", "rental revenue",
    "vacancy rates", "Halifax", "St. John's", "Ontario"
  ],
  "page_type": "body",
  "language": "en"
}
```

*Chi phí: $0.00524/trang | Input: 1,683 tokens | Output: 1,895 tokens | Total latency: 12,005ms*

Output trên minh họa bốn đặc điểm thực tế quan trọng của pipeline:

**(1) Lỗi OCR được giữ nguyên verbatim.** Các heading bị docTR nhận dạng sai do phông chữ in nghiêng và khoảng cách ký tự không đều trên trang scan — `"Manager m ent Discussi ion and Analysis"`, `"Apartmer nt ame Store NOI by City"`. LLM không tự sửa lỗi mà sao chép nguyên văn, đúng với thiết kế: giai đoạn LLM có vai trò cấu trúc hóa và tổng hợp, không phải correction. Điều này giúp downstream system phát hiện được những vùng OCR kém chất lượng để xử lý riêng.

**(2) Bảng số liệu được trích xuất chính xác hoàn toàn.** Mặc dù heading và paragraphs có lỗi OCR, bảng "Apartment Same Store NOI by City" với đầy đủ 5 cột và 9 hàng dữ liệu được trích xuất đúng 100% — bao gồm các giá trị âm trong ngoặc đơn như `"(72)"`, `"(1.1)%"`. Đây là kết quả của chiến lược định tuyến: Table block được gửi trực tiếp dưới dạng ảnh crop đến Gemini Vision thay vì qua docTR, bỏ qua hoàn toàn lỗi OCR của module text.

**(3) Page-footer lọt vào output.** Dòng cuối `"SILLAM ROP - - ERIES INC 2013 39"` là nội dung Page-footer (logo và số trang bị OCR nhiễu nặng). Pipeline hiện tại không lọc footer khỏi đầu vào LLM — đây là hạn chế đã được ghi nhận: một số Page-footer block có confidence đủ cao để vượt qua ngưỡng YOLO và được đưa vào prompt, gây nhiễu cho output.

**(4) Summary ngữ nghĩa chính xác dù input có lỗi.** Trường `summary` do Gemini tự tổng hợp mô tả đúng nội dung trang — bao gồm các con số cụ thể (NOI giảm 1.0%, St. John's và Ontario tăng 4.5%) — cho thấy LLM có khả năng phục hồi ngữ nghĩa từ văn bản OCR bị nhiễu ở mức độ ký tự, miễn là từ vẫn đọc được ở mức độ từ.

## 4.8 Benchmark hiệu năng và chi phí vận hành

Để có số liệu định lượng chính xác về tốc độ và chi phí vận hành, một benchmark riêng được thực hiện trên 50 mẫu chất lượng cao (chọn theo điểm composite từ kết quả eval — xem Phần 4.1.2). Benchmark đo wall-clock time từng module và ghi lại số liệu token sử dụng từ Gemini API.

### 4.8.1 Latency từng module

**Bảng 4.10: Latency pipeline trên 50 mẫu:**

| Module | Trung bình (ms) | Độ lệch chuẩn (ms) | Tỷ lệ tổng |
|--------|----------------|-------------------|------------|
| Preprocess (pdf2image + letterbox) | 134 | 51 | 1.7% |
| YOLO inference | 60 | 127 | 0.7% |
| docTR OCR (toàn bộ text blocks/trang) | 1,866 | 565 | 23.4% |
| XY-Cut reading order | <1 | <1 | <0.1% |
| LLM API call (Gemini 2.5 Flash) | 5,903 | 1,892 | 74.1% |
| **TOTAL pipeline** | **7,963** | **2,059** | 100% |

*Ghi chú: Model load time (YOLO: 6.5s, docTR: 4.6s) không tính vào per-sample latency. Độ lệch chuẩn cao của YOLO (127ms) do hiệu ứng warmup lần đầu.*

Kết quả cho thấy hai bottleneck rõ ràng: **LLM API call chiếm 74%** tổng thời gian và **OCR chiếm 23%**. YOLO inference (<1%) không phải nút thắt cổ chai dù đây là module phức tạp nhất về kiến trúc — phù hợp với thiết kế thời gian thực của YOLO. XY-Cut gần như không đáng kể (<1ms) nhờ độ phức tạp O(n log n) với n ≤ 30 block/trang.

Để cải thiện throughput trong triển khai thực tế: (1) OCR có thể song song hóa theo block; (2) LLM latency phụ thuộc infrastructure Gemini — không thể tối ưu ở phía client ngoài việc giảm prompt tokens.

### 4.8.2 Chi phí API Gemini

**Bảng 4.11: Thống kê token và chi phí Gemini 2.5 Flash (50 mẫu)**

| Chỉ số | Giá trị |
|--------|---------|
| Input tokens (mean) | 1,154 |
| Output tokens (mean) | 1,137 |
| Thinking tokens (mean) | 0 *(chế độ suy luận tắt)* |
| Chi phí/trang (mean) | **$0.00319** |
| Chi phí/1,000 trang | **~$3.19** |
| Schema parse rate | 100% (50/50) |
| Sections/trang (mean) | 3.1 |

Prompt yêu cầu sao chép nguyên văn (verbatim) làm output tokens tăng nhẹ so với phiên bản prompt cũ không có ràng buộc này (từ 1,117 lên 1,137 tokens, +1.8%), kéo chi phí tăng tương ứng từ $0.0031 lên $0.00319/trang (+2.5%). Mức tăng này không đáng kể trong khi độ đầy đủ nội dung cải thiện rõ rệt. Ước tính bảo thủ cho tập dữ liệu đa dạng nên dùng **$0.004/trang** (tương đương $4/1,000 trang).

### 4.8.3 So sánh chi phí với các giải pháp thay thế

**Bảng 4.12: So sánh chi phí xử lý tài liệu (USD/trang)**

| Giải pháp | Chi phí/trang | Ghi chú |
|-----------|-------------|---------|
| **Pipeline đề xuất** (YOLO+docTR+Gemini Flash) | **$0.0032–0.004** | Benchmark 50 mẫu; chế độ suy luận tắt |
| AWS Textract (text + table) | $0.015–0.020 | Pricing chính thức AWS (2024); chỉ OCR, không có layout |
| Google Document AI (Form Parser) | $0.065 | Pricing chính thức Google Cloud (2024) |
| Azure Document Intelligence (Layout) | $0.010 | Pricing chính thức Microsoft Azure (2024) |
| Gemini 2.5 Flash (thinking bật) | ~$0.018 | Ước tính với chế độ suy luận bật |
| Gemini 2.5 Pro | ~$0.060–0.080 | Ước tính với prompt tương tự |
| GPT-4o (với vision) | ~$0.025–0.040 | Ước tính dựa trên token pricing OpenAI (2024) |

*Lưu ý: Giá các dịch vụ thương mại có thể thay đổi; so sánh dựa trên pricing công bố tháng 5/2024. Chi phí pipeline đề xuất chưa bao gồm compute cost cho YOLO và docTR (chạy local/GPU).*

Pipeline đề xuất có chi phí thấp hơn đáng kể so với các dịch vụ thương mại tương đương. Lợi thế này đến từ ba yếu tố: (1) dùng Gemini 2.5 Flash thay vì Flash-thinking hay Pro; (2) tắt thinking mode; (3) strip COCO annotation data trước khi gửi prompt, giảm ~40% input tokens so với prompt nguyên bản.

Tuy nhiên, chi phí thấp đi kèm với bốn đánh đổi quan trọng:

- **Chất lượng suy luận:** Flash không có thinking mode đủ tốt cho tài liệu tài chính/pháp lý có cấu trúc rõ; tài liệu học thuật phức tạp (bảng lồng nhau, công thức, hình kỹ thuật) cần mô hình mạnh hơn (Gemini 2.5 Pro, GPT-4o) để phục hồi ngữ nghĩa từ OCR nhiễu.
- **Hạ tầng tự quản lý:** Pipeline yêu cầu duy trì GPU server cho YOLO và docTR; chi phí vận hành, quản lý GPU và cập nhật model không phản ánh trong con số $0.004/trang — TCO thực tế cao hơn đáng kể với tổ chức thiếu đội MLOps.
- **Khả năng mở rộng:** Ở quy mô triệu trang, chi phí Gemini API (~$4,000/1M trang) trở thành giới hạn; LLM local (Qwen2.5-7B, Llama 3.1-8B) có thể thay thế với chi phí gần zero sau đầu tư GPU ban đầu, dù chất lượng thấp hơn trên tài liệu phức tạp [17].
- **Hallucination:** Không có cơ chế hard-guarantee phát hiện khi LLM bịa nội dung từ OCR nhiễu; yêu cầu verbatim trong prompt chỉ là soft constraint [18].

### 4.8.4 So sánh chất lượng với các phương pháp baseline

Để đặt kết quả pipeline trong bối cảnh rộng hơn, một benchmark độc lập được thực hiện trên **100 mẫu chất lượng cao** chọn từ 267 ứng viên (điểm composite: content recall × 0.5 + det_f1 × 0.3 + CER × 0.2; lọc recall ≥ 0.75 và det_f1 ≥ 0.75). Ba phương pháp baseline được so sánh:

- **B1 — PyMuPDF/pdfplumber:** Trích xuất tầng text trực tiếp từ PDF, không phân tích bố cục.
- **B2 — docTR full-page OCR:** OCR toàn trang không qua YOLO detection, không phân tầng nội dung.
- **B3 — Gemini Vision full-page:** Gửi ảnh nguyên trang cho Gemini 2.5 Flash trích xuất văn bản plain-text (không có schema JSON cấu trúc của pipeline).

Hai chỉ số đo:
- *Content recall*: tỷ lệ token ground-truth xuất hiện trong output (fuzzy token matching).
- *Table token F1*: token F1 giữa nội dung bảng trong output và ground truth, đo trên 22 mẫu có bảng.

**Bảng 4.13: So sánh chất lượng pipeline với các phương pháp baseline (100 mẫu)**

| Phương pháp | n hợp lệ | Content Recall | Table Token F1 |
|---|---|---|---|
| **Pipeline đề xuất** | 99/100 | **0.986** | 0.939 |
| PyMuPDF/pdfplumber (B1) | 100/100 | 0.995 | 0.630 |
| docTR full-page OCR (B2) | 100/100 | 0.994 | 0.643 |
| Gemini Vision full-page (B3) | 89/100 | 0.992 | **0.981** |


**Nhận xét:**

**(1) Lợi thế rõ ràng ở Table F1.** Pipeline đạt Table F1 = 0.939, cao hơn PyMuPDF (0.630) và docTR full-page (0.643) khoảng **+30 điểm**. Đây là bằng chứng định lượng rõ ràng nhất rằng phân tích bố cục (YOLO detection → routing riêng cho block Table) mang lại giá trị so với trích xuất văn bản thuần túy. PyMuPDF và docTR đọc toàn trang mà không phân biệt block — nội dung bảng bị hòa lẫn vào text thông thường, làm giảm mạnh token F1 trên bảng.

**(2) Content recall: pipeline duy trì ở mức cạnh tranh.** PyMuPDF (0.995) và docTR (0.994) có content recall cao hơn pipeline (0.986) một lượng nhỏ (~0.009). Điều này hợp lý vì cả hai phương pháp đọc text layer đầy đủ mà không lọc hay tái cấu trúc, trong khi pipeline routing theo lớp có thể bỏ sót một số block bị YOLO miss. Khoảng cách nhỏ xác nhận rằng bước detection và routing không gây suy giảm đáng kể về độ bao phủ nội dung tổng thể.

**(3) Gemini Vision Table F1 = 0.981 cần diễn giải thận trọng.** Con số này cao hơn pipeline (0.939), tuy nhiên: (a) ground truth được tạo bởi Gemini 2.5 Flash (§4.1.2) — cả B3 lẫn GT đều chia sẻ cùng model prior, tạo circular bias đẩy F1 lên; (b) Gemini Vision có fail rate 11% (11/100 mẫu lỗi 503 UNAVAILABLE), không đảm bảo tính ổn định trong môi trường production.

**(4) Chi phí và độ tin cậy.** Pipeline ($0.0035/trang, fail rate 1%) có chi phí cao hơn Gemini Vision ($0.0025/trang) nhưng đổi lại độ tin cậy vượt trội — fail rate thấp hơn 11× và không phụ thuộc vào tính sẵn sàng của một API đơn lẻ. PyMuPDF và docTR có chi phí zero nhưng Table F1 thấp hơn ~30 điểm — phù hợp cho bài toán chỉ cần trích xuất text thuần, không yêu cầu cấu trúc bảng.

> Pipeline đề xuất là phương pháp duy nhất trong bốn phương án đạt đồng thời content recall ≥ 0.986 và table F1 ≥ 0.939 với fail rate thấp (1%). Lợi thế này đến từ layout-aware routing — tách biệt xử lý bảng/văn bản/tiêu đề thay vì đối xử đồng nhất toàn trang.

## 4.9 Tổng kết kết quả pipeline

**Bảng 4.14: Tổng kết kết quả toàn pipeline — các chỉ số chính**

| Module | Chỉ số chính | Giá trị |
|--------|-------------|---------|
| YOLO detection (§4.2) | mAP@0.5 (test) | **0.909** |
| YOLO detection (§4.2) | Lớp thấp nhất AP | 0.836 (Footnote) |
| Phát hiện khối end-to-end (§4.3) | F1 trung bình | **0.804** |
| Chất lượng OCR (§4.4) | CER Text/List-item | **~0.12** |
| LLM stage (§4.5) | Schema parse rate | **99.8%** |
| LLM stage (§4.5) | Content recall | **0.800** |
| LLM stage (§4.5) | Table token F1 | **0.870** |
| Hiệu năng | Total latency (mean) | 7,963 ms/trang |
| Hiệu năng | LLM bottleneck | 74% tổng latency |
| Chi phí | Gemini API | **~$0.0038/trang** |

> **Điểm mạnh cốt lõi:** Pipeline đạt balance tốt giữa độ chính xác, tốc độ và chi phí — không có một thành phần nào là nút thắt nghiêm trọng về chất lượng. Content recall 80% tiệm cận trần lý thuyết của dataset (~82–83% do annotation noise); chi phí thấp hơn 4–16× so với dịch vụ thương mại tương đương. So sánh trực tiếp với ba phương pháp baseline trên 100 mẫu (§4.8.4) cho thấy pipeline là phương án duy nhất đạt đồng thời content recall ≥ 0.986 và table token F1 ≥ 0.939.

---

# CHƯƠNG 5: KẾT LUẬN {#chương-5}

## 5.1 Tóm tắt kết quả đạt được

Trong luận văn này, em đã xây dựng và đánh giá một pipeline end-to-end cho bài toán phân tích bố cục và trích xuất cấu trúc ngữ nghĩa từ tài liệu PDF. Bốn kết quả chính đạt được như sau:

**Giai đoạn phát hiện bố cục:** YOLOv11s được huấn luyện trên DocLayNet với chiến lược oversampling đạt mAP@0.5 = 0.922 (kiểm định) và 0.909 (kiểm tra). Tất cả 11 lớp đều đạt AP > 0.83, kể cả các lớp cực hiếm như Title (0.47% dữ liệu) và Footnote (0.60%). Ablation study ba thí nghiệm cho thấy oversampling hiệu quả hơn augmentation mạnh trên tập dữ liệu tài liệu có spatial structure nhất quán.

**Giai đoạn OCR:** docTR được chọn qua benchmark ba engine trên 50 mẫu (§3.3.2), đạt CER ~0.12 trên lớp Text trong đánh giá đầy đủ 485 mẫu.

**Giai đoạn làm giàu ngữ nghĩa:** Gemini 2.5 Flash được tích hợp với chế độ suy luận tắt hoàn toàn, đạt schema parse rate 99.8%, content recall 0.800 và table token F1 0.870 trên 485 mẫu, với chi phí ~$0.0038/trang — thấp hơn 4–16× so với các dịch vụ thương mại tương đương (chi tiết tại §4.8). Benchmark so sánh độc lập trên 100 mẫu (§4.8.4) xác nhận pipeline vượt trội so với ba baseline (PyMuPDF, docTR full-page, Gemini Vision full-page) về table token F1 (+30 điểm so với trích xuất text thuần), trong khi content recall duy trì ở mức cạnh tranh (0.986).

**Framework đánh giá ba cấp:** Đánh giá tách biệt từng module trên 485 mẫu cho phép xác định đặc điểm hiệu suất và điểm yếu của từng thành phần, thay vì chỉ báo cáo kết quả tổng hợp cuối pipeline.

## 5.2 Đóng góp của luận văn

Luận văn tập trung vào tích hợp, đánh giá và phân tích một pipeline end-to-end cho document understanding — không đề xuất kiến trúc mới. Các đóng góp cụ thể:

- **Ablation xử lý mất cân bằng lớp (§4.2):** Với dữ liệu tài liệu có spatial structure ổn định (DocLayNet: 97:1 imbalance, 47.4% tiny bbox), oversampling cấp trang đủ hiệu quả mà không cần augmentation mạnh vốn phá vỡ spatial prior.
- **Framework đánh giá ba cấp + error budget (§4.5.1):** Tách biệt layout detection (F1 per-class), OCR (CER + coverage) và semantic enrichment (schema parse rate, content recall, table F1); xác định bốn nguồn thất thoát: detection miss (~10–12%), OCR noise (~4–6%), reading order/LLM (~3–5%), annotation noise (~2–3%). Bộ 485 mẫu GT và script benchmark có thể tái sử dụng.
- **Khảo sát trade-off chi phí–chất lượng và so sánh baseline (§4.8):** Truyền OCR text có reading order vào LLM thay vì ảnh nguyên trang đạt ~$0.0038/trang, thấp hơn 4–16× dịch vụ thương mại. Benchmark 100 mẫu (§4.8.4) định lượng lợi thế so với PyMuPDF, docTR full-page và Gemini Vision: pipeline là phương án duy nhất đạt table F1 ≥ 0.939 với fail rate ≤ 1%.
- **XY-Cut tùy chỉnh cho YOLO bbox:** Thuật toán reading order hai giai đoạn (mid-point ratio + vertical overlap), O(n log n), <1ms/trang, không cần mask segmentation.
- **Ghi nhận phương án bị loại:** PaddleOCR (xung đột cuDNN/DLL trên Windows, §3.3.1); Phi-3 Mini 4-bit (~923s/trang, chậm hơn 118× Gemini API, §3.5.2).

## 5.3 Kỹ năng và kiến thức đạt được

Trong quá trình thực hiện, em tích lũy được kinh nghiệm thực tiễn trên các khía cạnh sau:

**Kỹ năng kỹ thuật:**
- Huấn luyện và điều chỉnh mô hình YOLO trên tập dữ liệu tùy chỉnh: xử lý mất cân bằng lớp, thiết kế ablation study, phân tích kết quả định lượng từng lớp.
- Tích hợp nhiều model heterogeneous (YOLO, docTR, TATR, LLM) trong một pipeline end-to-end, quản lý VRAM trên GPU hạn chế (4GB) bằng chiến lược load/unload tuần tự.
- Xây dựng hệ thống benchmark tự động: thiết kế metrics đa cấp độ, viết script đánh giá pipeline với 485 mẫu, lưu trữ kết quả để so sánh.
- Thiết kế prompt engineering cho LLM: JSON schema enforcement, xử lý retry/fallback, đánh giá chất lượng output bằng token-level F1.
- Làm việc với thư viện deep learning trong môi trường thực tế: debug xung đột dependency (PaddleOCR/PyTorch), xử lý lỗi CUDA không nhất quán, tối ưu tốc độ inference.

**Kiến thức lý thuyết:**
- Nắm vững kiến trúc phát hiện đối tượng một giai đoạn (YOLO): anchor-free detection, multi-scale feature pyramid, loss function Varifocal Loss.
- Hiểu cơ chế OCR hiện đại: kiến trúc CRNN (CNN + BiLSTM + CTC), DBNet cho text detection, sự khác biệt giữa các engine về thiết kế và ứng dụng.
- Nắm lý thuyết và giới hạn của thuật toán reading order: XY-Cut và các biến thể, điểm yếu với layout tự do.
- Hiểu thực tiễn triển khai LLM: quantization 4-bit (BnB NF4), token budget management, trade-off giữa local inference và cloud API.
- Đánh giá khách quan mô hình: phân biệt validation vs test set, phân tích per-class vs macro, ý nghĩa của mAP, CER, F1 trong từng bài toán cụ thể.

**Kỹ năng nghiên cứu:**
- Đọc, trích lọc và tổng hợp thông tin từ tài liệu học thuật (paper, arXiv, kỷ yếu hội nghị).
- Tài liệu hóa quá trình thực nghiệm — bao gồm cả failure path (PaddleOCR, Phi-3 local) — để rút ra kết luận có căn cứ thay vì chỉ báo cáo kết quả tốt.
- Xác định giới hạn của phương pháp và đề xuất hướng cải thiện có tính khả thi kỹ thuật.

## 5.4 Đối chiếu với mục tiêu ban đầu

Ở §1.2, em đặt ra ba mục tiêu cụ thể: xây dựng pipeline end-to-end phát hiện bố cục và trích xuất cấu trúc, đánh giá định lượng từng module, và khảo sát thực nghiệm các lựa chọn kỹ thuật. Bảng sau tổng hợp mức độ đạt được:

| Mục tiêu | Kết quả | Đánh giá |
|---------|---------|---------|
| Pipeline end-to-end từ PDF đến JSON có cấu trúc | Triển khai đầy đủ: YOLO → docTR → XY-Cut → Gemini | Đạt |
| Đánh giá định lượng phát hiện bố cục | mAP@0.5=0.922 (val), 0.909 (test), F1 per-class trên 485 mẫu | Đạt |
| Đánh giá định lượng OCR | CER per-class, coverage | Đạt |
| Đánh giá định lượng LLM stage | Schema parse rate 99.8%, content recall 0.800, table F1 0.870 | Đạt |
| Khảo sát thực nghiệm OCR engine | Benchmark docTR/EasyOCR/TrOCR trên 50 mẫu | Đạt |
| Đánh giá định lượng reading order | Đánh giá gián tiếp + tập nhỏ 30 mẫu, chưa có metric tự động toàn tập | Đạt một phần |

Điểm em chưa đạt hoàn toàn là metric reading order tự động trên toàn tập 485 mẫu — DocLayNet không cung cấp ground truth thứ tự đọc nên em chỉ đánh giá được gián tiếp và trên tập nhỏ 30 mẫu. Đây là giới hạn của dữ liệu chuẩn hiện có, không phải của thiết kế thực nghiệm.

## 5.5 Hạn chế

- **Caption (F1=0.366) và Page-footer (F1=0.534):** Hai nguyên nhân chính: annotation không nhất quán trong DocLayNet (60% FN Page-footer và 45% FN Caption có thể quy về annotation noise, §4.6) và khó khăn thị giác cố hữu — Caption và Text chia sẻ đặc trưng hình ảnh giống nhau, chỉ phân biệt bởi vị trí tương đối với Figure/Table.
- **Phạm vi áp dụng hẹp:** Pipeline được đánh giá trên tài liệu PDF kỹ thuật số, tiếng Anh, bố cục 1–2 cột. Các điều kiện chưa kiểm chứng — scan nghiêng, DPI thấp, font đặc biệt, watermark, bảng không viền, tài liệu tiếng Việt — dự kiến gây suy giảm chất lượng đáng kể; kết quả hiện tại không nên ngoại suy sang các miền này.
- **Ngôn ngữ non-Latin:** docTR thiết kế cho Latin script. Với tiếng Việt, lỗi cascade OCR mất dấu → LLM nhận văn bản không dấu → content recall giảm mạnh; hiệu quả chưa được kiểm chứng.
- **Không phát hiện hallucination LLM:** Pipeline không có cơ chế phát hiện khi LLM fabricate nội dung không có trong OCR input hoặc thay thế block nhiễu bằng nội dung suy luận. Content recall không đủ nhạy để phân biệt hai trường hợp này (§4.5).
- **Context fragmentation:** Pipeline xử lý từng trang độc lập, không có ngữ cảnh từ trang trước — ảnh hưởng đến tài liệu có nội dung liên tục qua ranh giới trang.
- **Chi phí API và bảo mật:** ~$3,800/triệu trang và toàn bộ nội dung OCR được gửi ra hạ tầng bên thứ ba — cần cân nhắc với tài liệu nhạy cảm (§4.8, §5.6).
- **Reading order với layout phức tạp:** XY-Cut tùy chỉnh không xử lý tốt bố cục nhiều cột không đều hay text chạy vòng quanh hình.

## 5.6 Hướng phát triển

Các hướng dưới đây xuất phát trực tiếp từ hạn chế tại §5.5:

- **Sliding window context:** Đưa 3 block cuối trang trước vào context prefix, tăng ~3–5% input token nhưng giảm lỗi cắt đứt nội dung qua ranh giới trang.
- **Post-processing spatial Caption:** Reclassify block Text nằm ngay dưới Picture/Table (gap < 20px) thành Caption — không cần retrain.
- **Kiểm soát hallucination:** Tính n-gram coverage giữa OCR input và LLM output; flag trang dưới ngưỡng để review thủ công.
- **Prompt versioning + monitoring:** Theo dõi schema parse rate theo phiên bản prompt để phát hiện suy giảm chất lượng khi model API thay đổi.
- **Fine-tune docTR / VietOCR cho tiếng Việt:** Giải quyết lỗi cascade OCR mất dấu → LLM nhận văn bản không dấu.
- **LLM local (Qwen2.5-7B + QLoRA) on-premise [20]:** Thay Gemini để giảm chi phí ở quy mô triệu trang, hỗ trợ tiếng Việt và đáp ứng yêu cầu data governance (ngân hàng, y tế, pháp lý). Cần GPU ≥24GB.
- **Reading order học sâu thay XY-Cut:** Xử lý bố cục phức tạp nhiều cột không đều và text chạy vòng quanh hình.
- **Benchmark tập scan riêng (~50–100 trang):** Đánh giá hiệu suất trên ảnh scan thực tế (DPI thấp, nghiêng, nhiễu) — điều kiện chưa kiểm chứng trong thực nghiệm hiện tại.

---

# TÀI LIỆU THAM KHẢO {#tltk}

[1] B. Pfitzmann, C. Auer, M. Dolfi, A. S. Nassar, và P. W. J. Staar, "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis," trong *Proc. 28th ACM SIGKDD Conf. Knowledge Discovery and Data Mining (KDD '22)*, Washington DC, USA, 2022, pp. 3743–3751.

[2] X. Zhong, J. Tang, và A. J. Yepes, "PubLayNet: Largest Dataset ever for Document Layout Analysis," trong *Proc. Int. Conf. Document Analysis and Recognition (ICDAR)*, Sydney, Australia, 2019, pp. 1015–1022.

[3] M. Li, Y. Xu, L. Cui, S. Huang, F. Wei, Z. Li, và M. Zhou, "DocBank: A Benchmark Dataset for Document Layout Analysis," trong *Proc. 28th Int. Conf. Computational Linguistics (COLING)*, 2020, pp. 949–960.

[4] A. Khanam và M. Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements," *arXiv preprint arXiv:2410.17725*, 2024.

[5] C.-Y. Wang, I.-H. Yeh, và H.-Y. M. Liao, "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information," trong *Proc. European Conf. Computer Vision (ECCV)*, Milan, Italy, 2024.

[6] Z. Zhao, H. Yin, Z. Qi, J. Wang, Y. Zhao, và L. Li, "DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception," *arXiv preprint arXiv:2410.12628*, 2024.

[7] Y. Huang, T. Lv, L. Cui, Y. Lu, và F. Wei, "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking," trong *Proc. 30th ACM Int. Conf. Multimedia (ACM MM)*, Lisboa, Portugal, 2022, pp. 4083–4091.

[8] J. Li, Y. Xu, T. Lv, L. Cui, C. Zhang, và F. Wei, "DiT: Self-supervised Pre-training for Document Image Transformer," trong *Proc. 30th ACM Int. Conf. Multimedia (ACM MM)*, Lisboa, Portugal, 2022, pp. 3530–3539.

[9] H. Zhang, Y. Wang, F. Dayoub, và N. Sünderhauf, "VarifocalNet: An IoU-aware Dense Object Detector," trong *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 8514–8523.

[10] M. Guo, C. Yang, M. Zhang, P. Zhou, F. Yang, và G. Li, "Class Imbalance in Object Detection: An Experimental Diagnosis and Study of Mitigation Strategies," *arXiv preprint arXiv:2403.07113*, 2024.

[11] J. Li, X. Ma, W. Ye, T. Lu, và L. Wang, "RoDLA: Benchmarking the Robustness of Document Layout Analysis Models," trong *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 15759–15769.

[12] T. Pavlidis và J. Zhou, "Page Segmentation and Classification," *CVGIP: Graphical Models and Image Processing*, vol. 54, no. 6, pp. 484–496, 1992.

[13] R. Zhao và Z. Yin, "XY-Cut++: Advanced Document Layout Ordering via Hierarchical Mask Mechanism," *arXiv preprint arXiv:2504.10258*, 2025.

[14] A. Nguyen-Minh, D. Le-Duc, và T. Nguyen, "Reference-Based Post-OCR Processing with LLM for Diacritic Languages," *arXiv preprint arXiv:2410.13305*, 2024.

[15] W. Döhler, L. Goldmann, và A. Rauber, "OCR Error Post-Correction with LLMs: No Free Lunches," trong *Findings of the Assoc. Computational Linguistics (ACL)*, 2025.

[16] C. Auer, F. Pérez-García, L. Vogel, và J. Martins, "docTR: Document Text Recognition," *arXiv preprint arXiv:2307.07929*, 2023.

[17] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," *arXiv preprint arXiv:2307.09288*, 2023.

[18] Y. Zhang, Y. Li, L. Cui, D. Cai, L. Liu, T. Fu, X. Huang, E. Zhao, Y. Zhang, Y. Chen, L. Wang, A. Luu, W. Bi, F. Shi, và S. Shi, "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models," *arXiv preprint arXiv:2309.01219*, 2023.

[19] European Parliament and Council of the European Union, "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)," *Official Journal of the European Union*, L series, 2024. *(Quy định pháp lý về AI và xử lý dữ liệu nhạy cảm)*

[20] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, và W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," trong *Proc. Int. Conf. Learning Representations (ICLR)*, 2022.

[21] M. Liao, Z. Wan, C. Yao, K. Chen, và X. Bai, "Real-time Scene Text Detection with Differentiable Binarization," trong *Proc. 34th AAAI Conf. Artificial Intelligence (AAAI)*, New York, USA, 2020, pp. 11474–11481.

[22] B. Shi, X. Bai, và C. Yao, "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition," *IEEE Trans. Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 39, no. 11, pp. 2298–2304, 2017.

[23] M. Li, T. Lv, J. Chen, L. Cui, Y. Lu, D. Florencio, C. Zhang, Z. Li, và F. Wei, "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models," trong *Proc. 37th AAAI Conf. Artificial Intelligence (AAAI)*, Washington DC, USA, 2023, pp. 13094–13102.

[24] Google DeepMind, "Gemini 2.5 Flash," Google AI Developer Documentation, 2025. [Online]. Available: https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash

---

# PHỤ LỤC {#phu-luc}

## Phụ lục A: Cấu hình training đầy đủ (Thí nghiệm 2 — Oversampling)

```yaml
# ultralytics training config — Experiment 2 (Oversampling)
model: yolo11s.pt
data: dataset/BaseC/data_oversampled.yaml
imgsz: 640
batch: 24
epochs: 50
patience: 15
optimizer: SGD
lr0: 0.01
lrf: 0.01
cos_lr: true
nbs: 64
seed: 42
deterministic: true
rect: true
pretrained: true
val: true
save_period: 5
# Augmentation — conservative (Experiment 2)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
fliplr: 0.0      # KHÔNG flip ngang — tài liệu có spatial structure
mosaic: 1.0
close_mosaic: 10
```

## Phụ lục B: Ví dụ output JSON của pipeline

Xem Hình 4.12–4.13 và JSON output đầy đủ tại **§4.7** — bao gồm trang gốc, DocLayNet Ground Truth annotations, output JSON thực tế từ pipeline, và phân tích bốn đặc điểm quan trọng (lỗi OCR verbatim, trích xuất bảng chính xác, page-footer lọt vào output, khả năng phục hồi ngữ nghĩa của LLM).

## Phụ lục C: Kết quả benchmark OCR đầy đủ (50 mẫu DocLayNet)

Phân tích tổng hợp, tính nhất quán qua quy mô mẫu, và các trường hợp ngoại lệ đã được trình bày trong **§3.3.2**. Phụ lục này cung cấp bảng số liệu đầy đủ và ma trận quyết định cho mục đích tham khảo.

### C.1 Kết quả tổng hợp

| Engine | CER ↓ | WER ↓ | TextCov ↑ | Speed |
|--------|------:|------:|----------:|------:|
| **docTR v1.0.1** | **0.427** | **0.548** | 73.5% | 0.144 s/block |
| EasyOCR v1.7.2 | 0.448 | 0.615 | 74.4% | **0.130 s/block** |
| TrOCR (base-printed) | 0.706 | 0.836 | **98.0%** | 0.236 s/block |

*Phân tích tính nhất quán qua quy mô mẫu và các trường hợp EasyOCR vượt trội đã trình bày tại §3.3.2.*

### C.2 Ma trận quyết định chọn engine

| Tiêu chí | Trọng số | docTR | EasyOCR | TrOCR |
|---------|---------|-------|---------|-------|
| CER thấp | 30% | **0.427** — tốt nhất | 0.448 — tốt | 0.706 — kém (cắt xén văn bản) |
| WER thấp | 25% | **0.548** — tốt nhất | 0.615 — tốt | 0.836 — kém |
| Tốc độ (s/block) | 20% | 0.144 — nhanh | **0.130** — nhanh nhất | 0.236 — chậm |
| Text Coverage | 15% | 73.5% — trung bình | 74.4% — trung bình | **98.0%** — cao nhất* |
| Tương thích PyTorch | 10% | Có — cài đặt ổn định | Có — cài đặt ổn định | Có — cài đặt ổn định |
| **Điểm tổng hợp** | 100% | **4.45** | **4.05** | **2.65** |

*TextCov của TrOCR cao nhưng do đặc điểm cắt xén: cột chỉ đo từ có trong output, không phản ánh độ chính xác.

TrOCR bị loại do kiến trúc decoder chỉ nhận diện tối đa 21 từ mỗi dòng — phần lớn block Text và List-item trong DocLayNet dài hơn ngưỡng này, dẫn đến văn bản bị cắt ngắn có hệ thống. Đây là hạn chế thiết kế, không thể khắc phục mà không fine-tune lại mô hình.

---

## Phụ lục D: Đánh giá khả năng triển khai LLM offline — Phi-3 Mini 4-bit

Thử nghiệm đánh giá khả năng thay thế Gemini bằng LLM cục bộ để loại bỏ chi phí API và không gửi dữ liệu ra ngoài. Mô hình được chọn là **Phi-3.5-mini-instruct** (3.8B, 4-bit NF4, RTX 3050 Ti 4GB VRAM) — mô hình nhỏ nhất trong các lựa chọn khả thi (Qwen2.5-7B và Mistral-7B yêu cầu ~4.5GB, vượt VRAM).

**Kết quả so sánh trên 5 mẫu DocLayNet:**

| Metric | Phi-3 Mini 4-bit | Gemini 2.5 Flash |
|--------|:----------------:|:----------------:|
| Schema parse rate | 100% | 100% |
| Total latency (mean) | **945 s (~16 phút/trang)** | 8.0 s/trang |
| Sections/trang | 1.0 | 3.1 |
| Chi phí/trang | $0.00 | $0.0032 |
| Kết nối Internet | Không cần | Bắt buộc |

Phi-3 chậm hơn Gemini **118 lần** — không đáp ứng yêu cầu throughput thực tế. Chất lượng cũng thấp hơn: mô hình gom toàn bộ văn bản vào 1 section thay vì phân tách cấu trúc (3.1 sections/trang với Gemini). Latency dao động lớn (360s–2311s) tùy số block/trang.

Ba vấn đề kỹ thuật gặp phải: xung đột `rope_scaling` với transformers ≥5.0 (giải quyết bằng chuyển sang Phi-3.5), JSON bị cắt do `max_new_tokens` không đủ (tăng lên 2048), và VRAM không đủ khi load đồng thời các model (giải quyết bằng sequential load/unload).

**Ngưỡng GPU tối thiểu để LLM local khả thi (latency ≤30s/trang):**

| GPU | VRAM | Ước tính latency | Khả thi? |
|-----|------|-----------------|---------|
| RTX 3050 Ti | 4 GB | ~900 s | Không |
| RTX 4070 | 12 GB | ~15–30 s | Biên giới |
| RTX 4090 | 24 GB | ~5–10 s | Có |
| A100 | 40 GB | ~2–5 s | Có |

LLM local là hướng thay thế khả thi khi có GPU ≥24GB — phù hợp với yêu cầu bảo mật dữ liệu ở quy mô lớn (xem §5.6).

---

## Phụ lục E: Prompt LLM

Phụ lục này cung cấp nội dung chính xác của các prompt được dùng trong pipeline, tương ứng với mô tả thiết kế tại §3.5. Tất cả prompt được viết bằng tiếng Anh để phù hợp với ngôn ngữ của tập dữ liệu DocLayNet.

### E.1 Prompt chính — Schema JSON + Tóm tắt trang (Stage 4 / LLM call)

Đây là prompt production được dùng trong toàn bộ benchmark 485 mẫu (file `benchmark_pipeline.py`, hàm `assemble_and_summarize`). Prompt nhận danh sách block theo thứ tự đọc từ XY-Cut và trả về JSON có cấu trúc trong một API call duy nhất. Tương ứng với v2 compact (~350 tokens) trong Thí nghiệm A, §3.5.1.

Hai placeholder động được điền lúc runtime:
- `{layout_note}` — nếu trang 2 cột: thêm dòng `"LAYOUT NOTE: This page has a 2-COLUMN layout..."`, ngược lại để trống.
- `{table_note}` — nếu có table crops đính kèm: thêm dòng `"The N image(s) appended..."`, ngược lại để trống.
- `{blocks_text}` — nội dung block theo thứ tự đọc, mỗi dòng dạng `[ClassName] <văn bản>`.

```
You are a structured document analysis assistant.
Below are text blocks extracted from a single document page (in reading order).
{layout_note}{table_note}
Text blocks:
---
{blocks_text}
---

CRITICAL RULES:
1. Every [Section-header] block ALWAYS starts a NEW section, regardless of
   whether it has a number or not.
2. [Footnote] blocks belong to the section that physically contains them —
   copy them VERBATIM into footnotes[] of that section, NOT into paragraphs[].
3. [Page-header] and [Page-footer] blocks are page-level metadata — copy them
   VERBATIM into page_header[] and page_footer[] at the top level, NOT inside
   any section.
4. Copy each [Text], [Title], [List-item], [Caption] block VERBATIM into
   paragraphs[]. Do NOT summarize, paraphrase, shorten, or omit any block.
   Only the "summary" field should be a summary.

Return a single JSON object with this exact structure:
{
  "structured_json": {
    "title": "<string | null>",
    "page_header": ["<copy each Page-header block verbatim>"],
    "page_footer": ["<copy each Page-footer block verbatim>"],
    "sections": [
      {
        "heading": "<string | null — copy verbatim from Section-header block>",
        "paragraphs": ["<copy each Text/List-item/Caption block verbatim,
                         one string per block>"],
        "footnotes": ["<copy each Footnote block verbatim, one string per footnote>"],
        "tables": [{"headers": [], "rows": [[]]}],
        "formulas": ["<LaTeX>"],
        "figures": ["<description>"]
      }
    ]
  },
  "summary": "<3-5 sentence summary of the page>",
  "keywords": ["kw1", "kw2"],
  "page_type": "<title_page|table_of_contents|body|figure_page|reference|other>",
  "language": "<ISO 639-1>"
}

Return ONLY valid JSON. No markdown fences. No explanation.
```

**Ví dụ `{blocks_text}` thực tế:**

```
[Section-header] 3. Experimental Results
[Text] We evaluate our approach on three benchmark datasets...
[Table] (table crop image appended separately)
[Caption] Table 1: Comparison with state-of-the-art methods.
[Footnote] * Results reproduced from original paper.
[Page-footer] 12
```

**Schema validation:** Output được validate bằng Pydantic `PageSchema` (kiểm tra `sections` là list). Nếu parse fail, hệ thống retry tối đa 2 lần (tổng 3 lần). Parse rate trên 485 mẫu: 99.8% (484/485).

**Cấu hình Gemini:** `thinking_budget=0` — tắt hoàn toàn chain-of-thought reasoning. Lý do: xem §3.5.1 Thí nghiệm B.

---

