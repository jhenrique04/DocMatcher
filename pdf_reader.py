from pdfminer.high_level import extract_text


class PDFReader:
    @staticmethod
    def extract_text(file_path):
        return extract_text(file_path)
