from pydantic import BaseModel, Field

# Categorical codes are bounded to the values seen in training, since unseen
# codes become all-zero one-hot columns and produce silently wrong predictions.
PAY_STATUS = dict(ge=-2, le=8)

class CreditApplication(BaseModel):
    LIMIT_BAL: int = Field(gt=0)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY_0: int = Field(**PAY_STATUS)
    PAY_2: int = Field(**PAY_STATUS)
    PAY_3: int = Field(**PAY_STATUS)
    PAY_4: int = Field(**PAY_STATUS)
    PAY_5: int = Field(**PAY_STATUS)
    PAY_6: int = Field(**PAY_STATUS)
    BILL_AMT1: int
    BILL_AMT2: int
    BILL_AMT3: int
    BILL_AMT4: int
    BILL_AMT5: int
    BILL_AMT6: int
    PAY_AMT1: int = Field(ge=0)
    PAY_AMT2: int = Field(ge=0)
    PAY_AMT3: int = Field(ge=0)
    PAY_AMT4: int = Field(ge=0)
    PAY_AMT5: int = Field(ge=0)
    PAY_AMT6: int = Field(ge=0)
