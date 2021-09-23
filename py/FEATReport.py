from dataclasses import dataclass

from .FEATTests import FEATTest


@dataclass
class FEATReport:
    """FEAT test report to be generated in PDF, HTML, etc."""

    report_title: str
    report_desc: str
    feat_tests: list[FEATTest]
    # limitations: str = None
    # tradeoffs: str = None
    # ethical_considerations: str = None

    def run_tests(self) -> None:
        """Run tests that do not have 'result' or 'passed' attributes yet"""
        for ft in self.feat_tests:
            if ft.result is None or ft.passed is None:
                ft.run()

    def gen_appendix(self) -> dict[str, str]:
        """
        Return a dict of technique name to technique description for each
        FEAT test used.
        """
        return {ft.technique: ft.technique_desc for ft in self.feat_tests}
