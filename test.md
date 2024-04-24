! python ./src/train.py --lr 0.0001 --batch_size 16 --num_workers 16 --dropout 0.4 --ratio 0.5
0.53

! python ./src/train.py --lr 0.001 --batch_size 16 --num_workers 16 --dropout 0.4 --ratio 0.5
0.51

! python ./src/train.py --lr 0.001 --batch_size 64 --num_workers 64 --dropout 0.4 --ratio 0.5
NOT ENOUGH MEMORY

! python ./src/train.py --lr 0.001 --batch_size 64 --num_workers 8 --dropout 0.4 --ratio 0.5
NOT ENOUGH MEMORY

! python ./src/train.py --lr 0.001 --batch_size 32 --num_workers 8 --dropout 0.4 --ratio 0.5
0.49

! python ./src/train.py --lr 0.001 --batch_size 32 --num_workers 8 --dropout 0.4 --ratio 0.5 --num_filters 256
0.48

! python ./src/train.py --lr 0.001 --batch_size 32 --num_workers 16 --dropout 0.5 --num_epochs 100 --ratio 0.01


### MAMBA
! python ./src/train.py --lr 0.001 --batch_size 1 --num_workers 1 --dropout 0.5 --ratio 0.005 --mamba True
mamba works with this ^ but slow

! python ./src/train.py --lr 0.001 --batch_size 1 --num_workers 15 --dropout 0.5 --ratio 0.005 --mamba True
mambaworks with this too but slow

! python ./src/train.py --lr 0.001 --batch_size 4 --num_workers 15 --ratio 0.005 --mamba True
does not work memory overflow

