from enum import Enum

class LumiDType(Enum):
    int1 = 1
    int2 = 2
    int4 = 4
    int8 = 8
    int16 = 16
    int32 = 32
    int64 = 64
    fp4 = 104
    fp8 = 108
    fp16 = 116
    bf16 = 117
    fp32 = 132
    fp64 = 164