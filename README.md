# cnnaig
## preparation
 - place yosys executable to ~/yosys/yosys and abc executable to ~/abc/abc OR specify them as commandline arguments
 - if you are running on UNIX, the program will run cadex for verification (please chmod +x cadex), but it is just verification and not necessary
## run
python3 small.py dirname --train --quant --verilog
 - everything will be save in the directory "dirname"
 - --train option trains the model with FP weights
 - --quant option performs quantization of weights into power of two
 - --verilog generates verilog and blif files (call yosys and abc for synthesis)
## TODO
 - change model in small.py to get a relatively accurate AIG of at most 1,000,000 nodes
    - you can use only Conv2(same padding, ReLU), MaxPooling(valid padding, stride amount same as kernel size), AveragePooling, and Flatten&Dense
    - start from one conv layer, several pooling layers, and flatten&dense to understand good feature num and strides
    - later gradually replace pooling layers with conv layers
    - change epoch (option) to train for longer time
 - there are many commandline options, explore them after the model structure is somehow fixed
    - notice options are not thoroughly verified and may contain some bugs
