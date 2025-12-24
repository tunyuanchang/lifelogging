echo "=== CLIP Feature Extraction ==="
cd analysis
# python3 extract_openclip_image_features.py

echo "=== MongoDB Insertion ==="
cd ..
python3 new_combine.py output/video_fps.txt output/timestamp/ output/ocr/ output/asr/

# text db: caption
# python3 texts_mongodb.py