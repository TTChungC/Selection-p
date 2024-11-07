import datasets

id2label = {0:"activate_my_card",
1:"age_limit",
2:"apple_pay_or_google_pay",
3:"atm_support",
4:"automatic_top_up",
5:"balance_not_updated_after_bank_transfer",
6:"balance_not_updated_after_cheque_or_cash_deposit",
7:"beneficiary_not_allowed",
8:"cancel_transfer",
9:"card_about_to_expire",
10:"card_acceptance",
11:"card_arrival",
12:"card_delivery_estimate",
13:"card_linking",
14:"card_not_working",
15:"card_payment_fee_charged",
16:"card_payment_not_recognised",
17:"card_payment_wrong_exchange_rate",
18:"card_swallowed",
19:"cash_withdrawal_charge",
20:"cash_withdrawal_not_recognised",
21:"change_pin",
22:"compromised_card",
23:"contactless_not_working",
24:"country_support",
25:"declined_card_payment",
26:"declined_cash_withdrawal",
27:"declined_transfer",
28:"direct_debit_payment_not_recognised",
29:"disposable_card_limits",
30:"edit_personal_details",
31:"exchange_charge",
32:"exchange_rate",
33:"exchange_via_app",
34:"extra_charge_on_statement",
35:"failed_transfer",
36:"fiat_currency_support",
37:"get_disposable_virtual_card",
38:"get_physical_card",
39:"getting_spare_card",
40:"getting_virtual_card",
41:"lost_or_stolen_card",
42:"lost_or_stolen_phone",
43:"order_physical_card",
44:"passcode_forgotten",
45:"pending_card_payment",
46:"pending_cash_withdrawal",
47:"pending_top_up",
48:"pending_transfer",
49:"pin_blocked",
50:"receiving_money",
51:"Refund_not_showing_up",
52:"request_refund",
53:"reverted_card_payment?",
54:"supported_cards_and_currencies",
55:"terminate_account",
56:"top_up_by_bank_transfer_charge",
57:"top_up_by_card_charge",
58:"top_up_by_cash_or_cheque",
59:"top_up_failed",
60:"top_up_limits",
61:"top_up_reverted",
62:"topping_up_by_card",
63:"transaction_charged_twice",
64:"transfer_fee_charged",
65:"transfer_into_account",
66:"transfer_not_received_by_recipient",
67:"transfer_timing",
68:"unable_to_verify_identity",
69:"verify_my_identity",
70:"verify_source_of_funds",
71:"verify_top_up",
72:"virtual_card_not_working",
73:"visa_or_mastercard",
74:"why_verify_identity",
75:"wrong_amount_of_cash_received",
76:"wrong_exchange_rate_for_cash_withdrawal"}

def get_ds(dsname):
    if(dsname=='banking77'):
        train_dataset = datasets.load_dataset("legacy-datasets/banking77")["train"]
        test_dataset = datasets.load_dataset("legacy-datasets/banking77")["test"]
        options = list(id2label.values())
        template = "Context: {text}\nAnswer: {answer}"
        input_keys = ["text"]
        recalibrate_every = True
        balanced_sampling = True
        
    elif(dsname=='ag_news'):
        ag = datasets.load_dataset("ag_news")
        train_dataset = ag["train"]
        test_dataset = ag["test"]
        options = ["World", "Sports", "Business", "Sci/Tech"]
        template = "Article: {text}\nTopic: {answer}"
        input_keys = ["text"]
        recalibrate_every = False
        balanced_sampling = False
    elif(dsname=='copa'):
        train_dataset = datasets.load_dataset("super_glue", "copa")["validation"]
        test_dataset = datasets.load_dataset("super_glue", "copa")["train"]
        template = "Context: {premise}\nAnswer: {answer}"
        input_keys = ["premise"]
        recalibrate_every = True
        balanced_sampling = True
    elif(dsname=='sst2'):
        train_dataset = datasets.load_dataset("glue", "sst2")["train"]
        test_dataset = datasets.load_dataset("glue", "sst2")["validation"]
        options = ["negative", "positive"]
        template = "Sentence: {sentence}\nSentiment: {answer}"
        input_keys = ["sentence"]
        recalibrate_every = False
        balanced_sampling = True
    elif(dsname=='boolq'):
        train_dataset = datasets.load_dataset("super_glue", "boolq")["train"]
        test_dataset = datasets.load_dataset("super_glue", "boolq")["validation"]
        options = ["incorrect", "correct"]
        template = "{passage}\nquestion: {question}?\nanswer: {answer}"
        input_keys = ["passage"]
        recalibrate_every = True
        balanced_sampling = False
    elif(dsname=='wic'):
        train_dataset = datasets.load_dataset("super_glue", "wic")["train"]
        test_dataset = datasets.load_dataset("super_glue", "wic")["validation"]
        options = ["no", "yes"]
        template = "{sentence1}\n{sentence2}\nquestion: Is the word '{word}' used the same way in the two sentences above?\nanswer: {answer}"
        input_keys = ["sentence1", "sentence2"]
        recalibrate_every = True
        balanced_sampling = False
    elif(dsname=='wsc'):
        train_dataset = datasets.load_dataset("super_glue", "wsc")["train"]
        test_dataset = datasets.load_dataset("super_glue", "wsc")["validation"]
        options = ["no", "yes"]
        template = "Question: In the sentence \"{text}\", does the pronoun '{span2_text}' refer to {span1_text}?\nAnswer: {answer}"
        input_keys = ["text"]
        recalibrate_every = True
        balanced_sampling = False
    elif(dsname=='rte'):
        train_dataset = datasets.load_dataset("super_glue", "rte")["train"]
        test_dataset = datasets.load_dataset("super_glue", "rte")["validation"]
        options = ["True", "False"]
        template = "{premise}\nquestion: {hypothesis} True or False?\nanswer: {answer}"
        input_keys = ["hypothesis"]
        recalibrate_every = True
        balanced_sampling = False
    elif(dsname=='multirc'):
        train_dataset = datasets.load_dataset("super_glue", "multirc")["train"]
        test_dataset = datasets.load_dataset("super_glue", "multirc")["validation"]
        options = ["incorrect", "correct"]
        template = "Context: {paragraph}\n{question}\n{answer}\nanswer: {answer}"
        input_keys = ["paragraph"]
        recalibrate_every = True
        balanced_sampling = True
    elif(dsname=='subj'):
        train_dataset = datasets.load_dataset("SetFit/subj")["train"]
        test_dataset = datasets.load_dataset("SetFit/subj")["test"]
        options = ["objective", "subjective"]
        template = "input: {text}\ntype: {answer}"
        input_keys = ["text"]
        recalibrate_every = False
        balanced_sampling = True
    if(dsname=='copa'):
        train_dataset = train_dataset.map(lambda example: {**example, "options": [example["choice1"], example["choice2"]]})
        test_dataset = test_dataset.map(lambda example: {**example, "options": [example["choice1"], example["choice2"]]})
    else:
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})
    ds =  {
            "train": train_dataset,
            "test": test_dataset,
            "template": template,
            "input_keys": input_keys,
            "recalibrate_every": recalibrate_every,
            "balanced_sampling": balanced_sampling
        }
    return ds