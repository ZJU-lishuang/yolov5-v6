python train_sparsity.py --img 416 --batch 8 --epochs 50 --data data/coco_hand.yaml --cfg models/yolov5s.yaml --weights weights/last.pt --name s_hand_sparsity -sr --s 0.001 --prune 1

#python train_sparsity.py --img 416 --batch 8 --epochs 50 --data data/coco128.yaml --cfg models/yolov5s.yaml --weights weights/yolov5s.pt --name s_sparsity -sr --s 0.001 --prune 1