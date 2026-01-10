# ML-Weekly-task---Hands-On-
## 📚 Hands-on Machine Learning with Scikit-Learn, Keras, & TensorFlow - Summary

Rangkuman poin-poin kunci dari buku karya Aurélien Géron. Dokumen ini mencakup konsep dasar Machine Learning hingga penerapan Deep Learning skala besar.

---

## 🧠 Part I: The Fundamentals of Machine Learning

### Chapter 1: The Machine Learning Landscape
- ML adalah seni memprogram komputer untuk belajar dari data.
- Terbagi menjadi Supervised, Unsupervised, Semi-supervised, dan Reinforcement Learning.
- Tantangan utama: kekurangan data, data berkualitas buruk, dan overfitting/underfitting.
- Pentingnya memisahkan data menjadi Training Set dan Test Set.
- Validasi silang (Cross-validation) digunakan untuk evaluasi model yang lebih akurat.

### Chapter 2: End-to-End Machine Learning Project
- Memahami "Big Picture" dan tujuan bisnis sebelum mulai coding.
- Eksplorasi data (EDA) dan visualisasi untuk mendapatkan insight awal.
- Data Cleaning: menangani missing value, encoding kategori, dan scaling fitur.
- Memilih dan melatih berbagai model dasar (baseline).
- Fine-tuning hyperparameter menggunakan Grid Search atau Randomized Search.

### Chapter 3: Classification
- Klasifikasi biner vs multiclass vs multilabel.
- Metrik akurasi seringkali tidak cukup untuk dataset yang tidak seimbang (skewed).
- Gunakan Confusion Matrix, Precision, Recall, dan F1 Score.
- Kurva ROC dan AUC digunakan untuk membandingkan performa classifier.
- Error analysis membantu memahami di mana model melakukan kesalahan.

### Chapter 4: Training Models
- Linear Regression menggunakan persamaan Normal atau Gradient Descent.
- Gradient Descent: Batch, Stochastic (SGD), dan Mini-batch.
- Polynomial Regression untuk data non-linear (awas overfitting).
- Regularisasi (Ridge, Lasso, Elastic Net) untuk mengekang kompleksitas model.
- Logistic Regression dan Softmax Regression untuk klasifikasi probabilitas.

### Chapter 5: Support Vector Machines (SVM)
- Mencari hyperplane dengan margin terbesar (street) di antara kelas.
- Hard Margin (sensitif outlier) vs Soft Margin (lebih fleksibel).
- Kernel Trick (Polynomial/RBF) memetakan data ke dimensi tinggi tanpa komputasi berat.
- Efektif untuk dataset ukuran kecil hingga menengah yang kompleks.
- Dapat digunakan untuk klasifikasi, regresi, dan deteksi outlier.

### Chapter 6: Decision Trees
- Model "White Box" yang mudah diinterpretasikan dan divisualisasikan.
- Menggunakan algoritma CART (Classification and Regression Tree).
- Memisahkan node berdasarkan Gini Impurity atau Entropy.
- Cenderung overfitting, perlu regularisasi (max_depth, min_samples_leaf).
- Sensitif terhadap rotasi data dan variasi kecil dalam training set.

### Chapter 7: Ensemble Learning and Random Forests
- Menggabungkan banyak model (ensemble) seringkali lebih baik dari satu model terbaik.
- Voting Classifiers: Hard Voting (mayoritas) vs Soft Voting (probabilitas).
- Bagging (Bootstrap Aggregating) mengurangi variance (contoh: Random Forest).
- Boosting (AdaBoost, Gradient Boosting) melatih model secara sekuensial untuk memperbaiki error.
- Stacking melatih meta-model untuk menggabungkan prediksi dari model dasar.

### Chapter 8: Dimensionality Reduction
- "Curse of Dimensionality": data berdimensi tinggi itu sparse dan berisiko overfitting.
- Proyeksi (PCA) memampatkan data sambil menjaga varians (informasi) maksimal.
- Menentukan jumlah dimensi optimal menggunakan Explained Variance Ratio (misal 95%).
- Manifold Learning (LLE, t-SNE) efektif untuk data terpelintir (seperti Swiss Roll).
- Berguna untuk visualisasi data, kompresi, dan mempercepat training.

### Chapter 9: Unsupervised Learning Techniques
- K-Means Clustering mengelompokkan data berdasarkan centroid (perlu tahu jumlah k).
- Metode Elbow dan Silhouette Score untuk mencari jumlah cluster optimal.
- DBSCAN mendeteksi cluster berdasarkan kepadatan dan mengenali outlier.
- Gaussian Mixture Models (GMM) mengasumsikan data berasal dari distribusi Gaussian.
- Anomaly Detection mengidentifikasi instance langka di area kepadatan rendah.

---

## 🤖 Part II: Neural Networks and Deep Learning

### Chapter 10: Introduction to ANNs with Keras
- Perceptron hingga Multi-Layer Perceptron (MLP) untuk masalah non-linear.
- Backpropagation melatih jaringan menggunakan Gradient Descent.
- Sequential API untuk model sederhana, Functional API untuk topologi kompleks.
- Callbacks (EarlyStopping, ModelCheckpoint) untuk mengontrol proses training.
- Hyperparameter tuning otomatis menggunakan Keras Tuner.

### Chapter 11: Training Deep Neural Networks
- Masalah Vanishing/Exploding Gradients menghambat training jaringan dalam.
- Solusi: Inisialisasi bobot yang tepat (He/Glorot) dan fungsi aktivasi non-saturasi (ReLU/SELU).
- Batch Normalization menstabilkan dan mempercepat training secara drastis.
- Optimizer cepat: Momentum, RMSprop, dan Adam (standar industri).
- Regularisasi: Dropout, Alpha Dropout, dan Max-Norm untuk mencegah overfitting.

### Chapter 12: Custom Models and Training with TensorFlow
- Tensor (immutable) vs Variable (mutable) sebagai struktur data dasar.
- Membuat Custom Loss, Custom Metric, dan Custom Layer (subclassing).
- GradientTape memungkinkan Automatic Differentiation (autodiff) kustom.
- Custom Training Loop memberikan kontrol penuh atas proses optimasi.
- @tf.function mengubah fungsi Python menjadi Graf TensorFlow yang cepat.

### Chapter 13: Loading and Preprocessing Data
- tf.data API membangun pipeline input yang efisien (ETL).
- Prefetching memungkinkan GPU bekerja saat CPU menyiapkan batch berikutnya.
- Format TFRecord menggunakan Protocol Buffers untuk penyimpanan data biner besar.
- Keras Preprocessing Layers menanamkan logika pre-proses (normalisasi, encoding) ke dalam model.
- One-Hot Encoding vs Embeddings untuk fitur kategorikal.

### Chapter 14: Deep Computer Vision (CNNs)
- Convolutional Layers mengekstrak fitur visual lokal (tepi, tekstur).
- Pooling Layers (Max/Average) mengurangi dimensi spasial dan beban komputasi.
- Arsitektur modern: ResNet (Skip Connections), Xception, EfficientNet.
- Transfer Learning memanfaatkan model Pretrained (ImageNet) untuk tugas baru.
- Object Detection (YOLO) dan Semantic Segmentation untuk tugas visi tingkat lanjut.

### Chapter 15: Processing Sequences (RNNs & CNNs)
- RNN memproses data sekuensial dengan mempertahankan hidden state (memori).
- Masalah memori jangka pendek diatasi oleh LSTM dan GRU.
- Input shape berupa 3D tensor: [batch, time_steps, features].
- 1D CNN (WaveNet) bisa memproses sekuens panjang lebih cepat dari RNN.
- Aplikasi: Prediksi Time Series, Analisis Sentimen, dan Audio.

### Chapter 16: NLP with RNNs and Attention
- Character RNN bisa menghasilkan teks, tapi terbatas konteksnya.
- Word Embeddings menangkap makna semantik kata dalam vektor padat.
- Arsitektur Encoder-Decoder digunakan untuk Neural Machine Translation (NMT).
- Attention Mechanisms memecahkan bottleneck memori pada sekuens panjang.
- Transformer (BERT, GPT) menggantikan RNN dengan mekanisme Self-Attention murni.

### Chapter 17: Autoencoders and GANs
- Autoencoder mempelajari representasi padat (codings) secara unsupervised.
- Variational Autoencoders (VAE) bersifat generatif dan probabilistik.
- GANs: Generator membuat data palsu, Discriminator mencoba mendeteksinya.
- GANs sulit dilatih (Mode Collapse) tetapi menghasilkan gambar sangat realistis.
- Aplikasi: Super-resolution, Style Transfer, dan Anomaly Detection.

### Chapter 18: Reinforcement Learning
- Agen belajar melalui *trial and error* untuk memaksimalkan Reward.
- Policy Gradients (REINFORCE) mengoptimalkan neural network policy secara langsung.
- Q-Learning memprediksi nilai (quality) dari setiap aksi di suatu state.
- Deep Q-Networks (DQN) menggunakan Experience Replay untuk stabilitas.
- Algoritma canggih: Double DQN, Dueling DQN, dan Actor-Critic.

### Chapter 19: Training and Deploying at Scale
- TensorFlow Serving untuk menyajikan model di lingkungan produksi (REST/gRPC).
- TFLite untuk deploy model di perangkat mobile/edge (Android/IoT).
- Quantization mengurangi ukuran model dengan sedikit penurunan akurasi.
- Distributed Training: MirroredStrategy (Multi-GPU) dan TPUStrategy.
- Vertex AI (Google Cloud) untuk manajemen siklus hidup ML skala besar.

---

## 🏁 Kesimpulan

Buku ini memberikan panduan komprehensif mulai dari dasar-dasar Scikit-Learn hingga arsitektur Deep Learning mutakhir dengan TensorFlow 2. Kunci penguasaan materi ini adalah:
1.  **Pahami Datanya:** 80% pekerjaan adalah pembersihan dan penyiapan data.
2.  **Mulai Sederhana:** Gunakan model baseline sebelum mencoba arsitektur kompleks.
3.  **Hands-on:** Implementasi kode (terutama Custom Loops dan Subclassing) lebih penting daripada sekadar teori.
4.  **Skalabilitas:** Pikirkan bagaimana model akan di-deploy dan dijalankan di produksi sejak awal.
