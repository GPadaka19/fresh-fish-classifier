import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# 1. Load Model yang Sudah Disimpan
model_path = 'model/mac_fish_freshness_model.h5'
model = tf.keras.models.load_model(model_path)
print("✅ Model berhasil dimuat")

# 2. Daftar Kelas (Sesuaikan dengan model Anda)
class_names = ['Fresh_Eyes', 'Fresh_Gills', 'Nonfresh_Eyes', 'Nonfresh_Gills']

# 3. Fungsi Preprocessing Gambar
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Tambahkan batch dimension
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return img_array

# 4. Fungsi Prediksi
def predict_image(image_path):
    # Preprocess gambar
    processed_img = preprocess_image(image_path)
    
    # Lakukan prediksi
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # Tampilkan hasil
    plt.imshow(Image.open(image_path))
    plt.axis('off')
    plt.title(f"Prediksi: {class_names[predicted_class]}\nConfidence: {confidence:.2f}%")
    plt.show()
    
    return class_names[predicted_class], confidence

# 5. Interface Upload Sederhana
def main():
    print("\n🐟 Fish Freshness Classifier 🐟")
    print("="*30)
    
    while True:
        print("\nPilih opsi:")
        print("1. Test gambar dari path lokal")
        print("2. Keluar")
        choice = input("Masukkan pilihan (1/2): ")
        
        if choice == '1':
            image_path = input("Masukkan path gambar (contoh: 'test_fish.jpg'): ").strip('"')
            
            if not os.path.exists(image_path):
                print("❌ File tidak ditemukan!")
                continue
                
            try:
                class_name, confidence = predict_image(image_path)
                print(f"\nHasil Prediksi: {class_name} ({confidence:.2f}% confidence)")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif choice == '2':
            print("Terima kasih!")
            break
            
        else:
            print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()