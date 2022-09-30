export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
python -u Image_generator.py 1 6 > logs/partial_log1.log 2>&1 &
python -u Image_generator.py 2 6 > logs/partial_log2.log 2>&1 &
python -u Image_generator.py 3 6 > logs/partial_log3.log 2>&1 &
python -u Image_generator.py 4 6 > logs/partial_log4.log 2>&1 &
python -u Image_generator.py 5 6 > logs/partial_log5.log 2>&1 &
python -u Image_generator.py 6 6 > logs/partial_log6.log 2>&1 &