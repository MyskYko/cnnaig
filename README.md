# cnnaig
## preparation
 - place yosys executable to ~/yosys/yosys and abc executable to ~/abc/abc OR modify small.py to call them correctly
 - if you are running on UNIX, the program will run cadex for verification (please chmod +x cadex), but it is just verification and not necessary
## run
python3 small.py --train --quant --verilog
 - --train option trains the model with FP weights
 - --quant option performs quantization to power of two
 - --verilog generates verilog and blif files to test_(DATE) directory (call yosys and abc for synthesis)
## TODO
 - Change model and maxshamt in small.py to get a relatively accurate AIG of at most 1,000,000 nodes
    - You can use only Conv2(same padding, ReLU), MaxPooling(valid padding, stride amount same as kernel size), and Flatten&Dense
    - Change ntrainepoch to train longer time
 - Implement pruning
