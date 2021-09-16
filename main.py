# demo usage
from dataclasses import asdict
from pprint import pprint

from py.FEATTests import Sum, LIME
from py.FEATReport import FEATReport

# init tests
test1 = Sum(
    test_name='my first FEAT test',
    test_desc='',
    int1=1,
    int2=2
)

test2 = LIME(
    test_name='my LIME test',
    test_desc='sour',
    col1='hello',
    col2='bye'
)

# pass it into a report
report = FEATReport(
    report_title='my first FEAT report',
    report_desc='',
    feat_tests=[test1, test2]
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
