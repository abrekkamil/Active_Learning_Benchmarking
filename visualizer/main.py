import sys
from PyQt6.QtWidgets import QApplication
from gui import ExperimentGUI

app = QApplication(sys.argv)

window = ExperimentGUI()
window.show()

sys.exit(app.exec())