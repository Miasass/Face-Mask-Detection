# Face-Mask-Detection

### Baza danych
Do stworzenia projektu wykorzystana została [baza danych](https://www.kaggle.com/andrewmvd/face-mask-detection) z portalu Kaggle.com

### Prezentacja działania aplikacji
Działanie aplikacji zostało zaprezentowane na [filmie](https://youtu.be/jegcj527zGc)

### Jak załączyć aplikacje
Do uruchomienia aplikacji niezbędny jest Python 3.8 oraz biblioteki (cv2, numpy, keyboard, imutils, tensorflow).
Aby aplikacja zadziałała należy pobrać pliki znajdujące się [tutaj](https://drive.google.com/drive/folders/19z5TKETRr1-QXBMvhOhjVLsd-KHhRKDf?usp=sharing) i umieścić je w tym samym folderze, w którym znajduje się plik Application.ipynb.
Z pliku Application.ipynb należy uruchomić trzecią komórkę (dwie pierwsze można pominąć w przypadku, w którym posiada się wszystkie potrzebne biblioteki).


### Opis projektu
Celem projektu było stworzenie aplikacji wykrywającej jeden z trzech stanów (maska założona, maska nie założona oraz maska źle założona) w czasie rzeczywistym. Niestety z powodu zbyt małej liczby zdjęć prezentujących źle założoną maseczk, model nie wykrywa tego stanu.

Projekt składa się z trzech plików .ipynb
#### Data_Preprocessing.ipynb
W pliku tym znajdują się funkcje, dzięki którym zostały przetworzone dane do postaci umożliwiającej wprowadzenie ich do modelu sieci neuronowej.

Na samym początku wszystkie opisy zdjęć w postaci plików .xml dodane zostały do jednego obiektu Data Frame. Funkcja wykonująca to nosi nazwę "all_xml_to_df".
Następnie wykorzystując metadane ze zdjęć wycięte zostały fragmenty, na których znajdowała się twarz. Każdy z tych fragmentów zapisany został jako osobny obraz. Funkcja wykonująca to nosi nazwę "create_new_images".
Następnie dla każdej klasy (maska założona, maska nie założona oraz maska źle założona) obliczona została liczba obrazów przypadających na zbiór treningowy (80%), walidacyjny(10%) oraz testowy(10%). W dalszej kolejności obrazy zostały przekopiowane do odpowiednich folderów.

#### CNN_model.ipynb
W pliku tym zbudowane zostały dwa takie same modele konwolucyjnej sieci neuronowej. Do treningu pierwszego modelu wykorzystano dane nie poddane augmentacji. Natomiast do treningu drugiego modelu wykorzystano dane podddane augmentacji. Oba modele ze względu na zbyt małą liczbę obrazów prezentującyhc źle założoną maskę kompletnie nie były w stanie przewidzieć tej klasy. Modele uzyskały dokładnie taką samą dokładność 94,10%. Ze względu na uzyskanie takiej samej dokładności zdecydowano się na wykorzystanie w aplikacji wag dla pierwszego modelu.

#### Application.ipynb
W pliku tym znajduje się kod aplikacji. Dzięki bibliotece cv2 przechwytywany jest obraz z kamery. Następnie na obrazie przechwyconym z kamery wykrywana jest twarz/twarze. Z funkcji wykrywającej twarz uzyskane zostają współrzędne początku prostokąta oraz jego wysokość (prostokąt ten zakreśla twarz). Wycinek obrazu, na którym znajduje się twarz dostarczany jest do modelu sieci neuronowej, który wykrywa jeden z trzech stanów (maska założona, maska nie założona oraz maska źle założona). Następnie na podstawie klasy zwróconej przez model zmienia się kolor ramki obejmującej twarz oraz napis w górnym lewym rogu.
