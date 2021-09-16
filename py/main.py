# demo usage
from dataclasses import asdict
from pprint import pprint

from FEATTests import FEATTestSum, FEATTestLIME
from FEATReport import FEATReport

# init tests
test1 = FEATTestSum(
    test_name='my first FEAT test',
    test_desc='',
    int1=1,
    int2=2
)

test2 = FEATTestSum(
    test_name='my second FEAT test',
    test_desc='',
    int1=3,
    int2=4
)

test3 = FEATTestLIME(
    test_name='my LIME test',
    test_desc='sour',
    col1='hello',
    col2='bye'
)

# pass it into a report
report = FEATReport(
    report_title='my first FEAT report',
    report_desc='my first FEAT report consisting of 2 sum tests',
    feat_tests=[test1, test2, test3]
)

report.run_tests()

# pretty print report as a dict
# pprint(asdict(report), indent=2)
# print(test1)
# print(asdict(test1))
# print(test1.__dict__)

a = report.gen_appendix()
print(a)


# todo: pass report into pdf or html or something
