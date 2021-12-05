# python main.py --dataset wikics --embedder AFGRL --task node --layers [1024] --pred_hid 2048 --lr 0.001 --topk 8 --device 0 --mad 0.9
# python main.py --dataset computers --embedder AFGRL --task node --layers [512] --pred_hid 1024 --lr 0.001 --topk 4 --device 0 --mad 0.9
# python main.py --dataset photo --embedder AFGRL --task node --layers [512] --pred_hid 1024 --lr 0.001 --topk 4 --device 1 --mad 0.9
# python main.py --dataset cs --embedder AFGRL --task node --layers [1024] --pred_hid 2048 --lr 0.001 --topk 4 --device 1 --mad 0.9 --es 1000
# python main.py --dataset physics --embedder AFGRL --task node --layers [256] --pred_hid 512 --lr 0.01 --topk 8 --device 0 --mad 0.9 --es 1000

python main.py --dataset wikics --embedder AFGRL --task clustering --layers [1024] --pred_hid 2048 --lr 0.001 --topk 8 --device 0 --mad 0.9
python main.py --dataset computers --embedder AFGRL --task clustering --layers [512] --pred_hid 1024 --lr 0.001 --topk 4 --device 0 --mad 0.9
python main.py --dataset photo --embedder AFGRL --task clustering --layers [512] --pred_hid 1024 --lr 0.001 --topk 4 --device 0 --mad 0.9
python main.py --dataset cs --embedder AFGRL --task clustering --layers [1024] --pred_hid 2048 --lr 0.001 --topk 4 --device 0 --mad 0.9
python main.py --dataset physics --embedder AFGRL --task clustering --layers [256] --pred_hid 512 --lr 0.01 --topk 8 --device 0 --mad 0.9

# python main.py --dataset wikics --embedder AFGRL --task similarity --layers [1024] --pred_hid 2048 --lr 0.001 --topk 8 --device 0 --mad 0.9
# python main.py --dataset computers --embedder AFGRL --task similarity --layers [512] --pred_hid 1024 --lr 0.001 --topk 4 --device 0 --mad 0.9
# python main.py --dataset photo --embedder AFGRL --task similarity --layers [512] --pred_hid 1024 --lr 0.001 --topk 4 --device 1 --mad 0.9
# python main.py --dataset cs --embedder AFGRL --task similarity --layers [1024] --pred_hid 2048 --lr 0.001 --topk 4 --device 1 --mad 0.9 --es 1000
# python main.py --dataset physics --embedder AFGRL --task similarity --layers [256] --pred_hid 512 --lr 0.01 --topk 8 --device 0 --mad 0.9 --es 1000