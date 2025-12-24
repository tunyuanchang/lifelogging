
echo "=== Prepare All Contents ==="
# (head -n 1 *.csv && tail -n +2 -q *.csv) > all.csv

echo "=== Selective Saver ==="
python3 selective_saver.py -i ./output/si/loc1_script1_seq1_rec1/image_si.csv -c -1

echo "=== Downsampler ==="
# python3 remove.py
# [TODO] store keyframes in 'selected_images'