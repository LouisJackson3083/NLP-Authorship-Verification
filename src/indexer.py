# class INDEXER
# init function:
#   takes in vocab (just a list of targets in our case?), sets local variable vocab to it
#   initializes _vocab2idx and _idx2vocab to None
#   then finishes the vocab by turning it into a set (0 & 1's ????)
#
#   make build_vocab2idx function (stupid ass function)
#
#   make save function (writes to json)
#
#  
#
#
# class TOKENINDEXED
# init functions:
#   pass in itertools chain of FIRST_POS and SECOND_POS as vocabs
#   creates local vocabs using super INDEXER class
#   creates pad_index = 0, unknown_index = 1
#   initializes _vocab2idx and _idx2vocab to None
#
#   
# vocab2idx function:
#   builds a dictionary with pad and unknown indexes as 0 and 1
#   does whatever the fuck it does line 64
#
# idx2vocab function:
#   builds inverted dict of vocab2idx (it is literally the inverse lol)
#
#
#
#
#
#
#
#
#
#