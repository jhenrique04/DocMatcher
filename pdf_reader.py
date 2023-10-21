import PyPDF4

class PDFReader:
    def extract_text(self, file_path):
        with open(file_path, 'rb') as file:
            reader = PyPDF4.PdfFileReader(file)
            text = ""
            for page in range(reader.numPages):
                text += reader.getPage(page).extractText()
        return text