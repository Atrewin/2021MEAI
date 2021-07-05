def sortByNumber(phoneNumbers, index):
    buckets = [[]*9]
    for phoneNumber in phoneNumbers:
        indexNumber = phoneNumber/(10**index) % 10
        buckets[indexNumber].append(phoneNumber)
    i = 0
    for bucket in buckets:
        for phoneNumber in bucket:
            phoneNumbers[i] = phoneNumber
            i = i + 1
    return phoneNumbers

def sortPhoneNumbers(phoneNumbers):

    for index in range(11):
        phoneNumbers = sortByNumber(phoneNumbers, index)

    return phoneNumbers



if __name__ == "__main__":
    phoneNumbers = [12345678912,12345678913,16345678912,12315678912]
    a = sortPhoneNumbers(phoneNumbers)
    print(a)
    pass


def vaLR():
    pass
    # @改 除了bert 其他模块应该有更高的学习率 @改 0422
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # low_lr = ["sentence_encoder"]
    # # @jinhui 疑问点: 这里为什么要这样设置leanable parameters weight_decay
    # parameters_to_optimize = [
    #     {'params': [p for n, p in parameters_to_optimize
    #                 if (not any(nd in n for nd in no_decay)) and (any(low in n for low in low_lr))], 'weight_decay': 0.01},
    #     {'params': [p for n, p in parameters_to_optimize
    #                 if any(nd in n for nd in no_decay) and (any(low in n for low in low_lr))], 'weight_decay': 0.0},
    #     {'params': [p for n, p in parameters_to_optimize
    #                 if (not any(nd in n for nd in no_decay)) and ( not any(low in n for low in low_lr))],
    #      'weight_decay': 0.01, 'lr': 1e-1},# bert之外的人应该有更多的学习率
    #     {'params': [p for n, p in parameters_to_optimize
    #                 if any(nd in n for nd in no_decay) and (not any(low in n for low in low_lr))],
    #      'weight_decay': 0.0, 'lr': 1e-1}
    # ]
