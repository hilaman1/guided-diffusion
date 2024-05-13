# Train all versions of DiT - Polyps
python train.py --model-name DiT_S8_polyps --model DiT_S8 --data-path ./data/polyps --epochs 150 --cross-model false
python train.py --model-name DiT_S4_polyps --model DiT_S4 --data-path ./data/polyps --epochs 150 --cross-model false
python train.py --model-name DiT_S2_polyps --model DiT_S2 --data-path ./data/polyps --epochs 150 --cross-model false
python train.py --model-name DiT_B8_polyps --model DiT_B8 --data-path ./data/polyps --epochs 150 --cross-model false
python train.py --model-name DiT_B4_polyps --model DiT_B4 --data-path ./data/polyps --epochs 150 --cross-model false
python train.py --model-name DiT_B2_polyps --model DiT_B2 --data-path ./data/polyps --epochs 150 --cross-model false

# Train all versions of DiT cross - Polyps
python train.py --model-name DiT_S8_CROSS_polyps --model DiT_S8 --data-path ./data/polyps --epochs 150 --cross-model true
python train.py --model-name DiT_S4_CROSS_polyps --model DiT_S4 --data-path ./data/polyps --epochs 150 --cross-model true
python train.py --model-name DiT_S2_CROSS_polyps --model DiT_S2 --data-path ./data/polyps --epochs 150 --cross-model true
python train.py --model-name DiT_B8_CROSS_polyps --model DiT_B8 --data-path ./data/polyps --epochs 150 --cross-model true
python train.py --model-name DiT_B4_CROSS_polyps --model DiT_B4 --data-path ./data/polyps --epochs 150 --cross-model true
python train.py --model-name DiT_B2_CROSS_polyps --model DiT_B2 --data-path ./data/polyps --epochs 150 --cross-model true


# Train all versions of DiT - Kvasir
python train.py --model-name DiT_S8_Kvasir --model DiT_S8 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false
python train.py --model-name DiT_S4_Kvasir --model DiT_S4 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false
python train.py --model-name DiT_S2_Kvasir --model DiT_S2 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false
python train.py --model-name DiT_B8_Kvasir --model DiT_B8 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false
python train.py --model-name DiT_B4_Kvasir --model DiT_B4 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false
python train.py --model-name DiT_B2_Kvasir --model DiT_B2 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false

# Train all versions of DiT cross - Kvasir
python train.py --model-name DiT_S8_CROSS_Kvasir --model DiT_S8 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
python train.py --model-name DiT_S4_CROSS_Kvasir --model DiT_S4 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
python train.py --model-name DiT_S2_CROSS_Kvasir --model DiT_S2 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
python train.py --model-name DiT_B8_CROSS_Kvasir --model DiT_B8 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
python train.py --model-name DiT_B4_CROSS_Kvasir --model DiT_B4 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
python train.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
