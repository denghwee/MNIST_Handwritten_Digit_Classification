# Kiến trúc
Em triển khai mạng theo kiến trúc LeNet-5. Mạng sẽ có hai phần chính, bao gồm khối tích chập và khối kết nối đầy đủ:

![kiến trúc](https://media.discordapp.net/attachments/1276917701717266526/1360145212344762430/image.png?ex=67fa0d70&is=67f8bbf0&hm=4072136f058b3e6214655fa4940298cf0c2a4f41b09715c72b2ffa570d4b9cac&=&format=webp&quality=lossless&width=959&height=260)

Em triển khai trong phần code, net của em gồm các layer tương ứng như trên và mục đích của mỗi layer theo em tìm hiểu về LeNet:

- LazyConv2D(channels=6, kernel_size=5, padding=2) + sigmoid

  - Mục đích: trích xuất đặc trưng cơ bản từ ảnh đầu vào để tìm những đặc trưng sơ khai nhất, như cạnh, góc. Vì dùng một nhân 5x5 nên độ đệm sẽ đặt là 2 để giữ nguyên kích thước của ảnh. Cùng 6 kênh tương ứng đầu ra 6 features map, mỗi map sẽ học 1 kiểu đặc trưng

- AvgPool2D(pool_size=2, strides=2)

  - Mục đích: giảm kích thước không gian bằng kỹ thuật pooling (chiều rộng với chiều cao giảm một nửa)

- LazyConv2D(channels=16, kernel_size=5) + sigmoid

  - Mục đích: học các đặc trưng phức tạp từ các features map sau khi đã qua pooling. Ở đây em không thấy có padding trong kiến trúc, theo em tìm hiểu thì tác giả của kiến trúc muốn thu kẹp kích thước để tập trung vào trung tâm ảnh hơn, tránh hạn chế nhiễu viền. Số kênh tăng lên 16 để tăng lượng đặc trưng lên để làm mạng có lượng thông tin phong phú hơn

- AvgPool2D(pool_size=2, strides=2)

  - Mục đích: giảm kích thước không gian để trước khi chuyển sang phần mạng fully connected

Khối kết nối đầy đủ: (Trước khi vào mạng thì phải thực hiện flatten để chuyển tensor 4D thành vector 1D để đưa vào lớp fully connected)

- Linear(120) + sigmoid

- Linear(84) + sigmoid

- Linear(10) – đầu ra tương ứng với 10 lớp (chữ số 0–9)

  - Mục đích: Toàn bộ trên sẽ tương tự như một lớp học các đặc trưng được trích xuất ở lớp tích chập để tổng kết lại các đặc trưng được trích ra. Lớp trên em thấy nó sẽ tương tự y hệt như một mạng phân loại Softmax với đầu ra cuối cùng tương ứng 10 nhóm gồm 10 nhãn ký tự từ 0-1.

Trong đó sigmoid là để tạo tính phi tuyến, vì thời điểm mạng này tạo ra thì hàm kích hoạt relu chưa được phát triển ra, mà do dùng sigmoid nên điểm yếu của mạng trên là vanishing gradient.

- Để khắc phục điều đó, em sử dụng khởi tạo tham số với xavier distribution, mục đích là đảm bảo phương sai ở các đầu ra mỗi layer có phương sai bằng phương sai đầu vào để kiểm soát phương sai nên gradient sẽ không bị quá nhỏ, tránh hiện tượng gradient vanishing

Em giữ nguyên kiến trúc gần giống bản gốc LeNet-5, chỉ thay một các lớp Conv với Linear thành Lazy để tự động khớp input phù hợp.

# Triển khai

Dữ liệu ký tự MNIST được tải và tiền xử lý dưới dạng tensor 4 chiều: (batch_size, 1, 28, 28). Khác với đợt trước, thì bây giờ mình sẽ giữ nguyên tensor 4 chiều này mà không chuyển về dạng 2D bằng view nữa giữ nguyên 4 chiều để đẩy vào model luôn

Em sử dụng:

- Hàm mất mát: CrossEntropyLoss() - Vì đầu ra cuối cùng là bài toán Softmax

- Optimizer: SGD với learning rate = 0.1

- Epochs: 20 - Ban đầu em đặt 10 thì em thấy loss chỉ đến mức 0.4 và chưa bão hòa, nên em cảm thấy do lượng epochs quá ít nên model chưa học đủ. Xong em tăng lên khoảng epochs = 20 thì model bắt đầu có hiện tượng bắt đầu bão hòa ở các epoch thứ 19

- Batch size: 128 - em để ở mức này do em để các data chuyển sang gpu nên em giữ ở mức 128 thay vì cao hơn, tại em sợ khả năng gpu bị tràn bộ nhớ, do gpu em không quá khỏe, để đảm bảo an toàn trong lúc train, và em để số 128 để tối ưu hơn vì do nó là bội số của 8 (độ rộng của bit)

Kết quả cuối cùng thì loss là 0.208, sự khác biệt em nhận ra khi nhìn vào sơ đồ loss thì mô hình CNN LeNet sẽ ban đầu học rất chậm, lượng loss gần như không giảm có sự đáng kể và sau một vài epochs thì lượng loss giảm mạnh rõ rệt và sau một lượng epoch nhất định khác thì gradient bắt đầu bão hòa và loss giảm bắt đầu ít dần

![loss](https://media.discordapp.net/attachments/1276917701717266526/1360151897792712704/image.png?ex=67fa13aa&is=67f8c22a&hm=eeaa803523f4d971077b1b1fa3d2c0a755e49d23c877a3366fe9393d17161823&=&format=webp&quality=lossless&width=510&height=389)

Theo em hiểu thì nguyên nhân là do ban đầu model CNN trên lúc đầu khởi tạo với các tham số ngẫu nhiên, và model ở những epochs đầu đang cố tìm hiểu về những thông tin hữu ích trước khi thực sự bắt đầu học được điều gì đó ảnh hưởng lên điểm loss một cách đáng kể.
Các thông tin ở đây sẽ là hình dáng, các góc cạnh, đồng thời ban đầu gradients cũng đang rất nhỏ và khá hỗn tạp do model chưa trích xuất được nhiều thông tin từ train dataset.