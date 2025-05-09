#  GO-test: Phân loại ảnh chó/mèo bằng PyTorch và Streamlit

Dự án này bao gồm:
- Một notebook (`test.ipynb`) để huấn luyện mô hình phân loại chó/mèo sử dụng ResNet50 (PyTorch).
- Một ứng dụng web (`main.py`) dùng Streamlit để tải ảnh và dự đoán kết quả.

---

## Cài đặt thư viện

Chạy lệnh sau để cài đặt tất cả thư viện cần thiết:

```bash
pip install -r requirements.txt
```
## Huấn luyện mô hình (tạo checkpoint)

Mở và chạy **`test.ipynb`** để:

- Huấn luyện mô hình phân loại ảnh **chó/mèo** bằng ResNet50.
- Lưu lại mô hình đã huấn luyện vào file `checkpoint_epoch_4.pt`.

Bạn có thể dễ dàng chỉnh sửa số **epoch** hoặc thay đổi **dataset** trong notebook theo nhu cầu.




## Chạy giao diện phân loại ảnh

Sau khi đã có mô hình `checkpoint_epoch_4.pt`, bạn có thể chạy giao diện web bằng Streamlit:

```bash
streamlit run main.py
```

Sau đó mở trình duyệt và truy cập địa chỉ:
```bash
http://localhost:8501
```
### Tính năng:
Tải lên ảnh định dạng JPEG/PNG.

Nhận kết quả dự đoán ảnh là chó hay mèo.

### Web:
Bạn cũng có thể sử dụng web đã được deploy qua link sau:
```bash
https://go-test-ydemxjr3uvs8uykbmf2uof.streamlit.app/
```