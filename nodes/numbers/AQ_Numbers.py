import random

class AQ_Random:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                              "initial": ("INT", {"default": 0, "min": 0, "step": 1}),
                              }}
    RETURN_TYPES = ("INT",)
    FUNCTION = "generateRandomNumber"

    CATEGORY = "AQ/numbers"


    def generateRandomNumber(self, initial):
        randNumber = random.randint(0, 0xffffffffffffffff)

        return (randNumber,)


class AQ_Increment:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                              "initial": ("INT", {"default": 0, "min": 0, "step": 1}),
                              "force_initial": ("BOOLEAN", {"default": False}),
                              }}

    RETURN_TYPES = ("INT",)
    FUNCTION = "incrementNumber"

    CATEGORY = "AQ/numbers"

    numberFromPrevExecution = -1

    def incrementNumber(self, initial, force_initial):
        if(AQ_Increment.numberFromPrevExecution == -1 or force_initial):
            AQ_Increment.numberFromPrevExecution = initial
            return (initial,)

        currentState = AQ_Increment.numberFromPrevExecution
        AQ_Increment.numberFromPrevExecution += 1

        return (currentState,) 