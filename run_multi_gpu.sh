# # ----- Short Tests -----
# # python main.py 100 7e-5 1 0 &
# # python main.py 100 7e-5 1 1 &
# # python main.py 100 3e-5 1 2 &
# python main.py 1e-5 1 90 200 3 &
# python main.py 1e-5 1 90 200 4 &
# python main.py 1e-5 1 90 200 5 &
# python main.py 1e-5 1 90 200 6 &
# python main.py 1e-5 1 90 200 7 &

# ----- Long Tests -----
python main.py 1e-5 10 30 1000 3 -d xlarge -n many &
python main.py 1e-5 10 30 1000 4 -d xlarge -n many &
python main.py 1e-5 10 30 1000 5 -d xlarge -n many &
python main.py 1e-5 10 30 1000 6 -d xlarge -n many &
python main.py 1e-5 10 30 1000 7 -d xlarge -n many &