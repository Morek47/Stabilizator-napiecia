

#define NOMINMAX
#include <atomic>
#include <condition_variable>
#include <Eigen/Dense>
#include <windows.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <fstream>
#include <functional>
#include <dnnl.hpp>
#include <immintrin.h>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>
#include <filesystem>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/low_latency.hpp>
#include <openvino/pass/serialize.hpp>
#include <thread>
#include <mutex>
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <random> 
#include <memory>
#include <future>
#include <gdiplus.h>
#include <clang-c/Index.h>
#include <clang-c/Platform.h>
#include <regex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include "TrainingData.h"
#include <chrono>
#include <commctrl.h>
#include <cstdio>
#include <richedit.h>
#include <locale>
#include <codecvt>
#include <queue>



#include <iomanip>

#include <stack>

#include <system_error>

void LogToEditControl(HWND hwndEdit, const std::wstring& message) {
    int length = GetWindowTextLengthW(hwndEdit);
    SendMessage(hwndEdit, EM_SETSEL, (WPARAM)length, (LPARAM)length);
    SendMessage(hwndEdit, EM_REPLACESEL, 0, (LPARAM)message.c_str());
}


void updateErrorEditText(HWND hwndErrorEdit, const std::wstring& text) {
    SetWindowTextW(hwndErrorEdit, text.c_str());
}

void LogErrorToEditControl(HWND hwndEdit, const std::wstring& message) {
    int length = GetWindowTextLengthW(hwndEdit);
    SendMessage(hwndEdit, EM_SETSEL, (WPARAM)length, (LPARAM)length);
    SendMessage(hwndEdit, EM_REPLACESEL, 0, (LPARAM)message.c_str());
}
#define ID_EDIT_TEXT 101
#define WM_UPDATE_EDIT (WM_USER + 3)
#define WM_UPDATE_ERROR_EDIT (WM_USER + 4)
class RecurrentNeuralNetwork; // Deklaracja klasy

// Deklaracja zmiennych globalnych
std::atomic<bool> training_in_progress(true);
std::condition_variable cv;
std::mutex cv_mtx;
std::mutex mtx; // Mutex do synchronizacji
std::vector<double> loss_values;
std::vector<double> accuracy_values;
std::mutex clang_mutex;
std::atomic<bool> kod_zmieniony(false);
std::mutex kod_mtx;
std::mutex editTextMutex;
std::mutex codeMutex;
std::queue<std::string> message_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
std::atomic<bool> running(true);
std::thread messageThread;
std::mutex model_mutex;
std::shared_ptr<RecurrentNeuralNetwork> rnn;

HWND hwndEdit;
HWND hwndErrorEdit; // Dodano brakującą deklarację
HWND hMainWindow;



using namespace std;
using namespace Eigen;

using namespace Gdiplus;

#pragma comment(lib, "Gdiplus.lib")


// Definicja funkcji aktywacji
enum ActivationFunction {
    RELU,
    SIGMOID,
    TANH
};
using AlignedVectorXd = std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>>;
using AlignedMatrixXd = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;



// Funkcja do dodawania komunikatów do kolejki
void add_message(const std::string& message) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    message_queue.push(message);
    queue_cv.notify_one();
}


// Funkcja do pobierania komunikatów z kolejki
std::string get_message() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [] { return !message_queue.empty(); });
    std::string message = message_queue.front();
    message_queue.pop();
    return message;
}

void validate_index(int& index, int max_index) {
    if (index < 0) {
        index = 0;
    }
    else if (index >= max_index) {
        index = max_index - 1;
    }
}


std::wstring diagnostic_text = L"";
std::atomic<size_t> total_data_count(0);
std::atomic<size_t> auto_data_count(0);
std::atomic<size_t> manual_data_count(0);

void DrawDiagnosticText(HDC hdc, const RECT& rect, const std::wstring& text) {
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, RGB(0, 0, 0));
    DrawTextW(hdc, text.c_str(), -1, const_cast<RECT*>(&rect), DT_LEFT | DT_TOP | DT_WORDBREAK);
}

void DisplayTrainingResults(HWND hwnd) {
    std::wstring ws;
    ws.reserve(1024);

    ws.append(L"Wyniki treningu:\n");
    for (size_t i = 0; i < loss_values.size(); ++i) {
        ws.append(L"Epoka " + std::to_wstring(i + 1) + L": Loss = " + std::to_wstring(loss_values[i]) + L", Accuracy = " + std::to_wstring(accuracy_values[i]) + L"\n");
    }
    LogToEditControl(hwndErrorEdit, ws);
}

void check_alignment(const void* data, size_t alignment) {
    if (reinterpret_cast<uintptr_t>(data) % alignment != 0) {
        std::wstringstream ws;
        ws << L"Dane nie są wyrównane do " << alignment << L" bajtów!\n";
        LogErrorToEditControl(hwndErrorEdit, ws.str());
    }
    else {
        std::wstringstream ws;
        ws << L"Dane są wyrównane do " << alignment << L" bajtów.\n";
        LogToEditControl(hwndErrorEdit, ws.str());
    }
}




void check_alignment_32(const void* data) {
    check_alignment(data, 32);
}

__m128d ploadt(const double* data) {
    check_alignment(data, 16);
    return _mm_load_pd(data);
}

__m128d load(const double* data, int i, int j, int rows, int cols) {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Indeks poza zakresem");
    }
    return ploadt(&data[i * cols + j]);
}
void matrix_multiply_avx2(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N);


// Funkcje konwersji
std::string WideStringToString(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string str(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &str[0], size_needed, NULL, NULL);
    return str;
}

std::wstring StringToWideString(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size_needed);
    return wstr;
}


void example_usage() {
    vector<VectorXd> training_data = generuj_dane_treningowe(1000);

    try {
        double* data = training_data[0].data();
        int rows = training_data[0].rows();
        int cols = training_data[0].cols();
        load(data, 0, 0, rows, cols);
    }
    catch (const std::out_of_range& e) {
        LogErrorToEditControl(hwndErrorEdit, L"Indeks poza zakresem: ");
        LogErrorToEditControl(hwndErrorEdit, std::wstring(e.what(), e.what() + strlen(e.what())));
    }

    int N = 8;
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);

    matrix_multiply_avx2(A, B, C, N);

    std::wstringstream ws;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            ws << C[i * N + j] << " ";
        }
        ws << std::endl;
    }
    LogToEditControl(hwndErrorEdit, ws.str());

    std::wstring wstr = L"Przykładowy tekst";
    std::string str = WideStringToString(wstr);
    std::wstring converted_back = StringToWideString(str);

    std::wstringstream ws_conversion;
    ws_conversion << L"Original: " << wstr << std::endl;
    ws_conversion << L"Converted to string: " << str.c_str() << std::endl;
    ws_conversion << L"Converted back to wstring: " << converted_back << std::endl;
    LogToEditControl(hwndErrorEdit, ws_conversion.str());
}


// Przykładowe miejsce w kodzie, gdzie wykonywane jest mnożenie macierzy
void some_function() {
    int N = 8; // Przykładowy rozmiar macierzy
    std::vector<float> A(N * N, 1.0f); // Przykładowa macierz A
    std::vector<float> B(N * N, 1.0f); // Przykładowa macierz B
    std::vector<float> C(N * N, 0.0f); // Wynikowa macierz C

    // Zastąpienie istniejącej operacji mnożenia macierzy
    matrix_multiply_avx2(A, B, C, N);

    // Wyświetlenie wyników
    std::wstringstream ws;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            ws << C[i * N + j] << " ";
        }
        ws << std::endl;
    }
    LogToEditControl(hwndErrorEdit, ws.str());
}



vector<VectorXd> generuj_dane_treningowe(size_t liczba_wektorow) {
    vector<VectorXd> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < liczba_wektorow; ++i) {
        VectorXd vec(224 * 224 * 3);
        for (int j = 0; j < vec.size(); ++j) {
            validate_index(j, static_cast<int>(vec.size()));
            vec[j] = dis(gen);
        }
        // Sprawdzenie wyrównania pamięci
        check_alignment(vec.data(), 16);
        if (reinterpret_cast<uintptr_t>(vec.data()) % 16 != 0) {
            continue; // Pomijamy nieprawidłowe dane
        }
        data.push_back(vec);
        auto_data_count++;
        total_data_count++;
    }

    // Dodanie dodatkowych zróżnicowanych danych
    for (size_t i = 0; i < liczba_wektorow / 2; ++i) {
        VectorXd vec(224 * 224 * 3);
        for (int j = 0; j < vec.size(); ++j) {
            validate_index(j, static_cast<int>(vec.size()));
            vec[j] = dis(gen) * 2.0 - 1.0; // Wartości w zakresie [-1, 1]
        }
        // Sprawdzenie wyrównania pamięci
        check_alignment(vec.data(), 16);
        if (reinterpret_cast<uintptr_t>(vec.data()) % 16 != 0) {
            continue; // Pomijamy nieprawidłowe dane
        }
        data.push_back(vec);
        auto_data_count++;
        total_data_count++;
    }

    return data;
}

std::string kod_zrodlowy = R"(
#include <iostream>
void funkcja() {
    std::wstringstream ws;
    ws << L"Hello, World!" << std::endl;
    LogToEditControl(hwndErrorEdit, ws.str());
}
)";



// Struktura do przechowywania przypadków testowych
struct TestCase {
    std::string input;
    std::string expected_output;
};

// Funkcja do uruchamiania testów logicznych
double run_logical_tests(const std::string& modified_code) {
    // Zdefiniowanie przypadków testowych
    std::vector<TestCase> test_cases = {
        {"input1", "expected_output1"},
        {"input2", "expected_output2"},
        // Dodaj więcej przypadków testowych w razie potrzeby
    };

    // Zapisanie zmodyfikowanego kodu do pliku
    std::ofstream file("test_code.cpp");
    if (!file.is_open()) {
        updateErrorEditText(hwndErrorEdit, L"Nie można otworzyć pliku do zapisu testowego kodu.\n");
        return 0.0;
    }
    file << modified_code;
    file.close();

    // Kompilacja zmodyfikowanego kodu
    std::string compile_command = "g++ -o test_code.exe test_code.cpp";
    int compile_result = std::system(compile_command.c_str());
    if (compile_result != 0) {
        updateErrorEditText(hwndErrorEdit, L"Błąd kompilacji testowego kodu.\n");
        return 0.0;
    }

    int passed_tests = 0;

    // Uruchomienie testów
    for (const auto& test_case : test_cases) {
        // Uruchomienie skompilowanego kodu z danymi wejściowymi
        std::string run_command = "./test_code.exe " + test_case.input + " > output.txt";
        int run_result = std::system(run_command.c_str());
        if (run_result != 0) {
            updateErrorEditText(hwndErrorEdit, L"Błąd uruchomienia testowego kodu.\n");
            continue;
        }

        // Przechwycenie wyników działania kodu
        std::ifstream output_file("output.txt");
        if (!output_file.is_open()) {
            updateErrorEditText(hwndErrorEdit, L"Nie można otworzyć pliku z wynikami.\n");
            continue;
        }
        std::stringstream buffer;
        buffer << output_file.rdbuf();
        std::string output = buffer.str();
        output_file.close();

        // Porównanie uzyskanych wyników z oczekiwanymi
        if (output == test_case.expected_output) {
            passed_tests++;
        }
        else {
            std::wstringstream ws;
            ws << L"Test nie powiódł się. Oczekiwane: " << test_case.expected_output.c_str() << L", Uzyskane: " << output.c_str() << std::endl;
            updateErrorEditText(hwndErrorEdit, ws.str());
        }
    }

    // Obliczenie procentowego pokrycia testami logicznymi
    double coverage = static_cast<double>(passed_tests) / test_cases.size();
    return coverage;
}


void AddDataPoint(double loss, double accuracy) {
    std::lock_guard<std::mutex> lock(mtx);
    loss_values.push_back(loss);
    accuracy_values.push_back(accuracy);
}


void DrawGraph(HDC hdc, const RECT& rect, const std::vector<double>& data, const std::wstring& title) {
    if (data.empty()) return;

    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, RGB(0, 0, 0));
    TextOutW(hdc, rect.left + 10, rect.top + 10, title.c_str(), title.length());

    double max_value = *max_element(data.begin(), data.end());
    double min_value = *min_element(data.begin(), data.end());

    if (max_value == min_value) return;

    int graph_width = rect.right - rect.left;
    int graph_height = rect.bottom - rect.top;

    Rectangle(hdc, rect.left, rect.top, rect.right, rect.bottom);

    MoveToEx(hdc, rect.left, rect.bottom - static_cast<int>((data[0] - min_value) * graph_height / (max_value - min_value)), NULL);
    for (size_t i = 1; i < data.size(); ++i) {
        int x = rect.left + static_cast<int>(i * graph_width / static_cast<int>(data.size()));
        int y = rect.bottom - static_cast<int>((data[i] - min_value) * graph_height / (max_value - min_value));
        LineTo(hdc, x, y);
    }
}

// Funkcja aktywacji ReLU
double relu(double x) {
    return x > 0 ? x : 0;
}

// Pochodna funkcji aktywacji ReLU
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Funkcja aktywacji Sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Pochodna funkcji aktywacji Sigmoid
double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}

// Funkcja aktywacji Tanh
double tanh_activation(double x) {
    return tanh(x);
}

// Pochodna funkcji aktywacji Tanh
double tanh_derivative(double x) {
    double t = tanh(x);
    return 1 - t * t;
}





// Funkcja zwracająca odpowiednią funkcję aktywacji
std::function<double(double)> get_activation_function(ActivationFunction func) {
    switch (func) {
    case RELU:
        return relu;
    case SIGMOID:
        return sigmoid;
    case TANH:
        return tanh_activation;
    default:
        return relu;
    }
}

// Funkcja zwracająca odpowiednią pochodną funkcji aktywacji
std::function<double(double)> get_activation_derivative(ActivationFunction func) {
    switch (func) {
    case RELU:
        return relu_derivative;
    case SIGMOID:
        return sigmoid_derivative;
    case TANH:
        return tanh_derivative;
    default:
        return relu_derivative;
    }
}

void run_inference_on_npu(const std::string& model_xml_path, const std::string& model_bin_path, const std::vector<VectorXd>& inputs, HWND hwndErrorEdit) {
    try {
        ov::Core core;
        std::shared_ptr<ov::Model> model;

        try {
            model = core.read_model(model_xml_path, model_bin_path);
            LogToEditControl(hwndErrorEdit, L"Model pomyślnie wczytany.\n");
        }
        catch (const ov::Exception& e) {
            std::wstringstream ws;
            ws << L"Wyjątek OpenVINO podczas wczytywania modelu: " << e.what() << std::endl;
            LogErrorToEditControl(hwndErrorEdit, ws.str());
            return;
        }
        catch (const std::exception& e) {
            std::wstringstream ws;
            ws << L"Standardowy wyjątek podczas wczytywania modelu: " << e.what() << std::endl;
            LogErrorToEditControl(hwndErrorEdit, ws.str());
            return;
        }
        catch (...) {
            LogErrorToEditControl(hwndErrorEdit, L"Nieznany wyjątek podczas wczytywania modelu.\n");
            return;
        }

        ov::CompiledModel compiled_model;
        try {
            compiled_model = core.compile_model(model, "AUTO");
            LogToEditControl(hwndErrorEdit, L"Model pomyślnie skompilowany dla NPU.\n");
        }
        catch (const ov::Exception& e) {
            std::wstringstream ws;
            ws << L"Wyjątek OpenVINO podczas kompilacji modelu: " << e.what() << std::endl;
            LogErrorToEditControl(hwndErrorEdit, ws.str());
            return;
        }
        catch (const std::exception& e) {
            std::wstringstream ws;
            ws << L"Standardowy wyjątek podczas kompilacji modelu: " << e.what() << std::endl;
            LogErrorToEditControl(hwndErrorEdit, ws.str());
            return;
        }
        catch (...) {
            LogErrorToEditControl(hwndErrorEdit, L"Nieznany wyjątek podczas kompilacji modelu.\n");
            return;
        }

        ov::InferRequest infer_request;
        try {
            infer_request = compiled_model.create_infer_request();
            LogToEditControl(hwndErrorEdit, L"Żądanie inferencji pomyślnie utworzone.\n");
        }
        catch (const ov::Exception& e) {
            std::wstringstream ws;
            ws << L"Wyjątek OpenVINO podczas tworzenia żądania inferencji: " << e.what() << std::endl;
            LogErrorToEditControl(hwndErrorEdit, ws.str());
            return;
        }
        catch (const std::exception& e) {
            std::wstringstream ws;
            ws << L"Standardowy wyjątek podczas tworzenia żądania inferencji: " << e.what() << std::endl;
            LogErrorToEditControl(hwndErrorEdit, ws.str());
            return;
        }
        catch (...) {
            LogErrorToEditControl(hwndErrorEdit, L"Nieznany wyjątek podczas tworzenia żądania inferencji.\n");
            return;
        }

        const ov::Output<const ov::Node>& input_node = model->input();
        ov::Tensor input_tensor;
        try {
            input_tensor = infer_request.get_tensor(input_node);
        }
        catch (const ov::Exception& e) {
            std::wstringstream ws;
            ws << L"Wyjątek OpenVINO podczas przygotowywania tensora wejściowego: " << e.what() << std::endl;
            LogErrorToEditControl(hwndErrorEdit, ws.str());
            return;
        }
        catch (const std::exception& e) {
            std::wstringstream ws;
            ws << L"Standardowy wyjątek podczas przygotowywania tensora wejściowego: " << e.what() << std::endl;
            LogErrorToEditControl(hwndErrorEdit, ws.str());
            return;
        }
        catch (...) {
            LogErrorToEditControl(hwndErrorEdit, L"Nieznany wyjątek podczas przygotowywania tensora wejściowego.\n");
            return;
        }

        for (const auto& input : inputs) {
            if (input.size() * sizeof(float) > input_tensor.get_byte_size()) {
                LogErrorToEditControl(hwndErrorEdit, L"Rozmiar danych wejściowych przekracza rozmiar tensora wejściowego.\n");
                continue;
            }

            if (input.data() == nullptr) {
                LogErrorToEditControl(hwndErrorEdit, L"Wskaźnik input nie jest zainicjowany!\n");
                continue;
            }
            if (reinterpret_cast<uintptr_t>(input.data()) % 16 != 0) {
                LogErrorToEditControl(hwndErrorEdit, L"Dane wejściowe nie są wyrównane do 16 bajtów!\n");
                continue;
            }
            std::memcpy(input_tensor.data<float>(), input.data(), input.size() * sizeof(float));

            try {
                infer_request.infer();
                LogToEditControl(hwndErrorEdit, L"Inferencja pomyślnie wykonana.\n");
            }
            catch (const ov::Exception& e) {
                std::wstringstream ws;
                ws << L"Wyjątek OpenVINO podczas inferencji: " << e.what() << std::endl;
                LogErrorToEditControl(hwndErrorEdit, ws.str());
                return;
            }
            catch (const std::exception& e) {
                std::wstringstream ws;
                ws << L"Standardowy wyjątek podczas inferencji: " << e.what() << std::endl;
                LogErrorToEditControl(hwndErrorEdit, ws.str());
                return;
            }
            catch (...) {
                LogErrorToEditControl(hwndErrorEdit, L"Nieznany wyjątek podczas inferencji.\n");
                return;
            }

            const ov::Output<const ov::Node>& output_node = model->output();
            ov::Tensor output_tensor;
            try {
                output_tensor = infer_request.get_tensor(output_node);
            }
            catch (const ov::Exception& e) {
                std::wstringstream ws;
                ws << L"Wyjątek OpenVINO podczas pobierania tensora wyjściowego: " << e.what() << std::endl;
                LogErrorToEditControl(hwndErrorEdit, ws.str());
                return;
            }
            catch (const std::exception& e) {
                std::wstringstream ws;
                ws << L"Standardowy wyjątek podczas pobierania tensora wyjściowego: " << e.what() << std::endl;
                LogErrorToEditControl(hwndErrorEdit, ws.str());
                return;
            }
            catch (...) {
                LogErrorToEditControl(hwndErrorEdit, L"Nieznany wyjątek podczas pobierania tensora wyjściowego.\n");
                return;
            }

            try {
                const float* output_data = output_tensor.data<float>();
                std::vector<float> output_vector(output_data, output_data + output_tensor.get_size());
                std::wstringstream ws;
                ws << L"Wynik inferencji: ";
                for (const auto& val : output_vector) {
                    ws << val << L" ";
                }
                ws << std::endl;
                LogToEditControl(hwndErrorEdit, ws.str());
            }
            catch (const ov::Exception& e) {
                std::wstringstream ws;
                ws << L"Wyjątek OpenVINO podczas przetwarzania wyników inferencji: " << e.what() << std::endl;
                LogErrorToEditControl(hwndErrorEdit, ws.str());
                return;
            }
            catch (const std::exception& e) {
                std::wstringstream ws;
                ws << L"Standardowy wyjątek podczas przetwarzania wyników inferencji: " << e.what() << std::endl;
                LogErrorToEditControl(hwndErrorEdit, ws.str());
                return;
            }
            catch (...) {
                LogErrorToEditControl(hwndErrorEdit, L"Nieznany wyjątek podczas przetwarzania wyników inferencji.\n");
                return;
            }
        }
    }
    catch (const std::exception& e) {
        std::wstringstream ws;
        ws << L"Standardowy wyjątek: " << e.what() << std::endl;
        LogErrorToEditControl(hwndErrorEdit, ws.str());
    }
    catch (...) {
        LogErrorToEditControl(hwndErrorEdit, L"Nieznany wyjątek.\n");
    }
}

// Użycie aligned_allocator dla wektorów i macierzy
using AlignedVectorXd = std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>>;
using AlignedMatrixXd = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;

// Struktura optymalizatora Adam
struct AdamOptimizer {
    double learning_rate; // Współczynnik uczenia
    double beta1; // Parametr beta1
    double beta2; // Parametr beta2
    double epsilon; // Parametr epsilon
    AlignedMatrixXd m; // Wektory m
    AlignedMatrixXd v; // Wektory v
    int t; // Licznik iteracji

    // Konstruktor optymalizatora Adam
    AdamOptimizer(double lr, double b1, double b2, double eps, const AlignedMatrixXd& weights)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {
        for (const auto& w : weights) {
            m.push_back(Eigen::MatrixXd::Zero(w.rows(), w.cols()));
            v.push_back(Eigen::MatrixXd::Zero(w.rows(), w.cols()));
        }
    }

    // Funkcja aktualizująca wagi
    void update(AlignedMatrixXd& weights, const AlignedMatrixXd& gradients) {
        t++;
        for (size_t i = 0; i < weights.size(); ++i) {
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1 - beta2) * gradients[i].cwiseProduct(gradients[i]);

            Eigen::MatrixXd m_hat = m[i] / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v[i] / (1 - pow(beta2, t));

            weights[i] -= learning_rate * m_hat.array().cwiseQuotient(v_hat.array().sqrt() + epsilon).matrix();
        }
    }
};
class RecurrentNeuralNetwork {
public:
    // Konstruktor przyjmujący cztery argumenty
    RecurrentNeuralNetwork(int input_size, int hidden_size, int output_size, ActivationFunction activation_func)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size), activation_func(activation_func) {
        Wxh = std::make_unique<MatrixXd>(MatrixXd::Random(hidden_size, input_size));
        Whh = std::make_unique<MatrixXd>(MatrixXd::Random(hidden_size, hidden_size));
        Why = std::make_unique<MatrixXd>(MatrixXd::Random(output_size, hidden_size));
        bh = std::make_unique<VectorXd>(VectorXd::Random(hidden_size));
        by = std::make_unique<VectorXd>(VectorXd::Random(output_size));
        h = std::make_unique<VectorXd>(VectorXd::Zero(hidden_size)); // Inicjalizacja wektora stanu ukrytego

        // Dodanie dodatkowych warstw ukrytych
        Whh2 = std::make_unique<MatrixXd>(MatrixXd::Random(hidden_size, hidden_size));
        bh2 = std::make_unique<VectorXd>(VectorXd::Random(hidden_size));

        // Sprawdzenie wymiarów macierzy i wektorów
        std::cout << "Wxh dimensions: " << Wxh->rows() << "x" << Wxh->cols() << std::endl;
        std::cout << "Whh dimensions: " << Whh->rows() << "x" << Whh->cols() << std::endl;
        std::cout << "Why dimensions: " << Why->rows() << "x" << Why->cols() << std::endl;
        std::cout << "bh size: " << bh->size() << std::endl;
        std::cout << "by size: " << by->size() << std::endl;
        std::cout << "h size: " << h->size() << std::endl;
    }

    // Nowy konstruktor przyjmujący trzy argumenty
    RecurrentNeuralNetwork(int input_size, int hidden_size, int output_size)
        : RecurrentNeuralNetwork(input_size, hidden_size, output_size, ActivationFunction::RELU) {
    }

    VectorXd forward(const VectorXd& input) {
        std::wstringstream ws;
        ws << L"Rozpoczęcie funkcji forward, rozmiar wejścia: " << input.size() << std::endl;
        LogToEditControl(hwndErrorEdit, ws.str());

        if (input.size() != input_size) {
            throw std::invalid_argument("Rozmiar wejścia nie jest zgodny z oczekiwanym rozmiarem " + std::to_string(input_size) + ".");
        }

        if (input.size() != Wxh->cols()) {
            throw std::invalid_argument("Rozmiar wejścia nie jest zgodny z rozmiarem macierzy Wxh.");
        }

        auto activation = get_activation_function(activation_func);
        for (int i = 0; i < input.size(); ++i) {
            validate_index(i, input.size());
        }

        check_alignment(input.data(), 16);
        check_alignment(Wxh->data(), 16);
        check_alignment(Whh->data(), 16);
        check_alignment(bh->data(), 16);

        if (input.data() == nullptr || Wxh->data() == nullptr || Whh->data() == nullptr || bh->data() == nullptr) {
            throw std::runtime_error("Wskaźnik danych jest nullptr.");
        }

        if (Wxh->rows() != hidden_size || Whh->rows() != hidden_size || Why->rows() != output_size) {
            throw std::runtime_error("Rozmiar macierzy nie jest zgodny z oczekiwanym rozmiarem.");
        }

        *h = (*Wxh * input) + (*Whh * *h) + *bh;
        *h = h->unaryExpr(activation);

        *h = (*Whh2 * *h) + *bh2;
        *h = h->unaryExpr(activation);

        VectorXd output = (*Why * *h) + *by;
        ws.str(L"");
        ws << L"Zakończenie funkcji forward, rozmiar wyjścia: " << output.size() << std::endl;
        LogToEditControl(hwndErrorEdit, ws.str());
        return output;
    }

    void backward(const VectorXd& input, const VectorXd& target, double learning_rate) {
        VectorXd output = forward(input);
        VectorXd error = target - output;

        MatrixXd dWhy = error * h->transpose();
        VectorXd dby = error;
        auto activation_derivative = get_activation_derivative(activation_func);
        VectorXd dh = Why->transpose() * error + h->unaryExpr(activation_derivative);

        *Why += learning_rate * dWhy;
        *by += learning_rate * dby;

        // Obliczenia gradientów dla Whh2 i bh2
        VectorXd dh2 = Whh2->transpose() * dh;
        dh2 = dh2.cwiseProduct(h->unaryExpr(activation_derivative));

        *Whh2 += learning_rate * (dh2 * h->transpose());
        *bh2 += learning_rate * dh2;

        for (int i = 0; i < input.size(); ++i) {
            validate_index(i, input.size());
        }
        *Wxh += learning_rate * (dh * input.transpose());
        *Whh += learning_rate * (dh * h->transpose());
        *bh += learning_rate * dh;
    }

    

    void train(const vector<VectorXd>& inputs, const vector<VectorXd>& targets, double learning_rate) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            backward(inputs[i], targets[i], learning_rate);
        }
    }

    // Funkcja zapisu modelu do pliku
    void save_model(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Nie można otworzyć pliku do zapisu: " << filename << std::endl;
            return;
        }

        // Zapisujemy wymiary macierzy i wektorów
        file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
        file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

        // Zapisujemy wagi i biasy
        save_matrix(file, *Wxh);
        save_matrix(file, *Whh);
        save_matrix(file, *Why);
        save_vector(file, *bh);
        save_vector(file, *by);

        file.close();
    }

    // Funkcja wczytywania modelu z pliku
    void load_model(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Nie można otworzyć pliku do odczytu: " << filename << std::endl;
            return;
        }

        // Wczytujemy wymiary macierzy i wektorów
        file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
        file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

        // Wczytujemy wagi i biasy
        load_matrix(file, *Wxh);
        load_matrix(file, *Whh);
        load_matrix(file, *Why);
        load_vector(file, *bh);
        load_vector(file, *by);

        file.close();

        // Wywołanie run_inference_on_npu po wczytaniu modelu
        std::vector<VectorXd> dummy_inputs = generuj_dane_treningowe(1); // Przykładowe dane wejściowe
        run_inference_on_npu("model.xml", "model.bin", dummy_inputs, hwndErrorEdit);
    }

    // Gettery do prywatnych elementów składowych
    int get_input_size() const {
        return input_size;
    }

    int get_output_size() const {
        return output_size;
    }

    // Dodanie metody get_model
    std::shared_ptr<ov::Model> get_model() const {
        // Tworzenie modelu OpenVINO
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 1, static_cast<size_t>(input_size) });
        std::shared_ptr<ov::Node> current = input;

        // Dodawanie warstw do modelu
        auto Wxh_const = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape{ static_cast<size_t>(hidden_size), static_cast<size_t>(input_size) }, Wxh->data());
        auto Whh_const = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape{ static_cast<size_t>(hidden_size), static_cast<size_t>(hidden_size) }, Whh->data());
        auto bh_const = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape{ static_cast<size_t>(hidden_size) }, bh->data());
        auto Why_const = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape{ static_cast<size_t>(output_size), static_cast<size_t>(hidden_size) }, Why->data());
        auto by_const = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape{ static_cast<size_t>(output_size) }, by->data());

        auto Wx_input = std::make_shared<ov::opset8::MatMul>(current, Wxh_const);
        auto Wh_hidden = std::make_shared<ov::opset8::MatMul>(current, Whh_const);
        auto hidden_add = std::make_shared<ov::opset8::Add>(Wx_input, Wh_hidden);
        auto hidden_bias = std::make_shared<ov::opset8::Add>(hidden_add, bh_const);
        auto hidden_activation = std::make_shared<ov::opset8::Tanh>(hidden_bias);

        auto output_matmul = std::make_shared<ov::opset8::MatMul>(hidden_activation, Why_const);
        auto output_add = std::make_shared<ov::opset8::Add>(output_matmul, by_const);

        auto result = std::make_shared<ov::opset8::Result>(output_add);
        auto function = std::make_shared<ov::Model>(ov::ResultVector{ result }, ov::ParameterVector{ input });

        return function;
    }

private:
    int input_size, hidden_size, output_size;
    ActivationFunction activation_func;
    std::shared_ptr<MatrixXd> Wxh, Whh, Whh2, Why;
    std::shared_ptr<VectorXd> bh, bh2, by, h;

    // Funkcja pomocnicza do zapisu macierzy
    void save_matrix(std::ofstream& file, const MatrixXd& matrix) {
        long rows = static_cast<long>(matrix.rows()); // Dodano rzutowanie
        long cols = static_cast<long>(matrix.cols()); // Dodano rzutowanie
        file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        file.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(double));
    }

    // Funkcja pomocnicza do zapisu wektorów
    void save_vector(std::ofstream& file, const VectorXd& vector) {
        long size = static_cast<long>(vector.size()); // Dodano rzutowanie
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(vector.data()), size * sizeof(double));
    }

    // Funkcja pomocnicza do wczytywania macierzy
    void load_matrix(std::ifstream& file, MatrixXd& matrix) {
        long rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        matrix.resize(rows, cols);
        file.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(double));
    }

    // Funkcja pomocnicza do wczytywania wektorów
    void load_vector(std::ifstream& file, VectorXd& vector) {
        long size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vector.resize(size);
        file.read(reinterpret_cast<char*>(vector.data()), size * sizeof(double));
    }
};

// Klasa głębokiej sieci neuronowej
class DeepNeuralNetwork {
public:
    // Konstruktor sieci neuronowej
    DeepNeuralNetwork(const std::vector<int>& layers, ActivationFunction activation_func)
        : activation_func(activation_func) {
        srand(static_cast<unsigned int>(time(0)));
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            weights.push_back(Eigen::MatrixXd::Random(layers[i + 1], layers[i]));
            biases.push_back(Eigen::VectorXd::Random(layers[i + 1]));
        }
    }

    // Getter for weights
    const AlignedMatrixXd& get_weights() const {
        return weights;
    }

    // Getter for biases
    const AlignedVectorXd& get_biases() const {
        return biases;
    }

    // Funkcja trenująca sieć neuronową z regularyzacją L1
    void train_with_l1_regularization(const AlignedVectorXd& batch_inputs, const AlignedVectorXd& batch_targets, double learning_rate, AdamOptimizer& optimizer, double lambda) {
        AlignedMatrixXd weight_gradients(weights.size());
        AlignedVectorXd bias_gradients(biases.size());

        for (size_t i = 0; i < weights.size(); ++i) {
            weight_gradients[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
            bias_gradients[i] = Eigen::VectorXd::Zero(biases[i].size());
        }

        for (size_t b = 0; b < batch_inputs.size(); ++b) {
            if (b >= batch_inputs.size() || b >= batch_targets.size()) {
                std::wstringstream ws;
                ws << L"Indeks poza zakresem w batch_inputs lub batch_targets: " << b << std::endl;
                LogErrorToEditControl(hwndErrorEdit, ws.str());
                continue;
            }

            std::vector<Eigen::VectorXd> activations = feedforward(batch_inputs[b]);
            Eigen::VectorXd output_errors = batch_targets[b] - activations.back();

            for (int i = static_cast<int>(weights.size()) - 1; i >= 0; --i) {
                if (i >= activations.size()) {
                    std::wstringstream ws;
                    ws << L"Indeks poza zakresem w activations: " << i << std::endl;
                    LogErrorToEditControl(hwndErrorEdit, ws.str());
                    continue;
                }

                Eigen::VectorXd delta = output_errors.cwiseProduct(activations[i + 1].unaryExpr(get_activation_derivative(activation_func)));
                weight_gradients[i] += delta * activations[i].transpose();
                bias_gradients[i] += delta;
                if (i > 0) {
                    output_errors = weights[i].transpose() * output_errors;
                }
            }
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            weight_gradients[i] /= static_cast<double>(batch_inputs.size());
            bias_gradients[i] /= static_cast<double>(batch_inputs.size());
            weight_gradients[i] += lambda * weights[i].array().sign().matrix(); // Dodanie regularyzacji L1
        }

        optimizer.update(weights, weight_gradients);
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] -= learning_rate * bias_gradients[i];
        }
    }

    // Funkcja trenująca sieć neuronową z regularyzacją L2
    void train_with_l2_regularization(const AlignedVectorXd& batch_inputs, const AlignedVectorXd& batch_targets, double learning_rate, AdamOptimizer& optimizer, double lambda) {
        AlignedMatrixXd weight_gradients(weights.size());
        AlignedVectorXd bias_gradients(biases.size());

        for (size_t i = 0; i < weights.size(); ++i) {
            weight_gradients[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
            bias_gradients[i] = Eigen::VectorXd::Zero(biases[i].size());
        }

        for (size_t b = 0; b < batch_inputs.size(); ++b) {
            if (b >= batch_inputs.size() || b >= batch_targets.size()) {
                std::wstringstream ws;
                ws << L"Indeks poza zakresem w batch_inputs lub batch_targets: " << b << std::endl;
                LogErrorToEditControl(hwndErrorEdit, ws.str());
                continue;
            }

            std::vector<Eigen::VectorXd> activations = feedforward(batch_inputs[b]);
            Eigen::VectorXd output_errors = batch_targets[b] - activations.back();

            for (int i = static_cast<int>(weights.size()) - 1; i >= 0; --i) {
                if (i >= activations.size()) {
                    std::wstringstream ws;
                    ws << L"Indeks poza zakresem w activations: " << i << std::endl;
                    LogErrorToEditControl(hwndErrorEdit, ws.str());
                    continue;
                }

                Eigen::VectorXd delta = output_errors.cwiseProduct(activations[i + 1].unaryExpr(get_activation_derivative(activation_func)));
                weight_gradients[i] += delta * activations[i].transpose();
                bias_gradients[i] += delta;
                if (i > 0) {
                    output_errors = weights[i].transpose() * output_errors;
                }
            }
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            weight_gradients[i] /= static_cast<double>(batch_inputs.size());
            bias_gradients[i] /= static_cast<double>(batch_inputs.size());
            weight_gradients[i] += lambda * weights[i]; // Dodanie regularyzacji L2
        }

        optimizer.update(weights, weight_gradients);
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] -= learning_rate * bias_gradients[i];
        }
    }

   
    // Funkcja trenująca sieć neuronową z wczesnym zatrzymaniem
    void train_with_early_stopping(const AlignedVectorXd& train_inputs, const AlignedVectorXd& train_targets,
        const AlignedVectorXd& val_inputs, const AlignedVectorXd& val_targets,
        double learning_rate, AdamOptimizer& optimizer, double lambda,
        int max_epochs, int patience) {
        int epochs_no_improve = 0;
        double best_val_loss = std::numeric_limits<double>::infinity();

        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            // Trening na zbiorze treningowym z regularyzacją L1
            train_with_l1_regularization(train_inputs, train_targets, learning_rate, optimizer, lambda);

            // Trening na zbiorze treningowym z regularyzacją L2
            train_with_l2_regularization(train_inputs, train_targets, learning_rate, optimizer, lambda);

            // Obliczanie straty na zbiorze walidacyjnym
            double val_loss = compute_loss(val_inputs, val_targets);

            std::wstringstream ws;
            ws << L"Epoch " << epoch + 1 << L", Validation Loss: " << val_loss << std::endl;
            LogToEditControl(hwndErrorEdit, ws.str());

            // Sprawdzanie, czy strata na zbiorze walidacyjnym się poprawiła
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                epochs_no_improve = 0;
            }
            else {
                epochs_no_improve++;
            }

            // Sprawdzanie, czy należy zatrzymać trening
            if (epochs_no_improve >= patience) {
                ws.str(L"");
                ws << L"Early stopping at epoch " << epoch + 1 << std::endl;
                LogToEditControl(hwndErrorEdit, ws.str());
                break;
            }
        }
    }

private:
    AlignedMatrixXd weights; // Wagi sieci
    AlignedVectorXd biases; // Biasy sieci
    ActivationFunction activation_func; // Funkcja aktywacji

    // Funkcja propagacji w przód
    std::vector<Eigen::VectorXd> feedforward(const Eigen::VectorXd& inputs) {
        std::vector<Eigen::VectorXd> activations;
        activations.push_back(inputs);
        for (size_t i = 0; i < weights.size(); ++i) {
            Eigen::VectorXd z = weights[i] * activations.back() + biases[i];
            activations.push_back(z.unaryExpr(get_activation_function(activation_func)));
        }
        return activations;
    }

    // Funkcja obliczająca stratę
    double compute_loss(const AlignedVectorXd& inputs, const AlignedVectorXd& targets) {
        double loss = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Eigen::VectorXd output = feedforward(inputs[i]).back();
            loss += (targets[i] - output).squaredNorm();
        }
        return loss / inputs.size();
    }
};

// Funkcja dzieląca dane na zbiory treningowy, walidacyjny i testowy
void split_data(const AlignedVectorXd& inputs, const AlignedVectorXd& targets,
    AlignedVectorXd& train_inputs, AlignedVectorXd& train_targets,
    AlignedVectorXd& val_inputs, AlignedVectorXd& val_targets,
    AlignedVectorXd& test_inputs, AlignedVectorXd& test_targets) {
    size_t total_size = inputs.size();
    double train_ratio = 0.7;
    double val_ratio = 0.15;
    if (total_size > 1000) {
        train_ratio = 0.6;
        val_ratio = 0.2;
    }
    size_t train_size = static_cast<size_t>(total_size * train_ratio);
    size_t val_size = static_cast<size_t>(total_size * val_ratio);

    std::vector<size_t> indices(total_size);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < train_size; ++i) {
        train_inputs.push_back(inputs[indices[i]]);
        train_targets.push_back(targets[indices[i]]);
    }

    for (size_t i = train_size; i < train_size + val_size; ++i) {
        val_inputs.push_back(inputs[indices[i]]);
        val_targets.push_back(targets[indices[i]]);
    }

    for (size_t i = train_size + val_size; i < total_size; ++i) {
        test_inputs.push_back(inputs[indices[i]]);
        test_targets.push_back(targets[indices[i]]);
    }


    // Przykład zastąpienia std::cout funkcją LogToEditControl
    std::wstringstream ws;
    ws << L"Podział danych zakończony. Całkowity rozmiar: " << total_size << L", Rozmiar treningowy: " << train_size << L", Rozmiar walidacyjny: " << val_size << L", Rozmiar testowy: " << (total_size - train_size - val_size) << L"\n";
    LogToEditControl(hwndErrorEdit, ws.str());
}



void load_cnn_model(const std::string& model_xml_path, const std::string& model_bin_path, ov::Core& core, std::shared_ptr<ov::Model>& model) {
    try {
        model = core.read_model(model_xml_path, model_bin_path);
        std::wstringstream ws;
        ws << L"Model CNN pomyślnie wczytany z " << std::wstring(model_xml_path.begin(), model_xml_path.end()) << L" i " << std::wstring(model_bin_path.begin(), model_bin_path.end()) << L"\n";
        LogToEditControl(hwndErrorEdit, ws.str());
    }
    catch (const ov::Exception& e) {
        std::wstringstream ws;
        ws << L"Wyjątek OpenVINO podczas wczytywania modelu: " << std::wstring(e.what(), e.what() + strlen(e.what())) << L"\n";
        updateErrorEditText(hwndErrorEdit, ws.str());
    }
    catch (const std::exception& e) {
        std::wstringstream ws;
        ws << L"Standardowy wyjątek podczas wczytywania modelu: " << std::wstring(e.what(), e.what() + strlen(e.what())) << L"\n";
        updateErrorEditText(hwndErrorEdit, ws.str());
    }
    catch (...) {
        updateErrorEditText(hwndErrorEdit, L"Nieznany wyjątek podczas wczytywania modelu\n");
    }
}

void run_cnn_inference(const std::shared_ptr<ov::Model>& model, const std::vector<float>& input_data, std::vector<float>& output_data) {
    try {
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model, "AUTO");
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        const ov::Output<const ov::Node>& input_node = model->input();
        ov::Tensor input_tensor = infer_request.get_tensor(input_node);
        std::memcpy(input_tensor.data<float>(), input_data.data(), input_data.size() * sizeof(float));

        infer_request.infer();

        const ov::Output<const ov::Node>& output_node = model->output();
        ov::Tensor output_tensor = infer_request.get_tensor(output_node);
        output_data.resize(output_tensor.get_size());
        std::memcpy(output_data.data(), output_tensor.data<float>(), output_tensor.get_size() * sizeof(float));
    }
    catch (const ov::Exception& e) {
        std::wstringstream ws;
        ws << L"Wyjątek OpenVINO podczas inferencji: " << std::wstring(e.what(), e.what() + strlen(e.what())) << L"\n";
        updateErrorEditText(hwndErrorEdit, ws.str());
    }
    catch (const std::exception& e) {
        std::wstringstream ws;
        ws << L"Standardowy wyjątek podczas inferencji: " << std::wstring(e.what(), e.what() + strlen(e.what())) << L"\n";
        updateErrorEditText(hwndErrorEdit, ws.str());
    }
    catch (...) {
        updateErrorEditText(hwndErrorEdit, L"Nieznany wyjątek podczas inferencji\n");
    }
}


void optimize_model_with_openvino(const std::string& model_xml_path, const std::string& model_bin_path) {
    try {
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_xml_path, model_bin_path);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::LowLatency2>();
        manager.register_pass<ov::pass::Serialize>(model_xml_path, model_bin_path);
        manager.run_passes(model);
        std::wstringstream ws;
        ws << L"Model został zoptymalizowany przy użyciu OpenVINO.\n";
        LogToEditControl(hwndErrorEdit, ws.str());
    }
    catch (const ov::Exception& e) {
        std::wstringstream ws;
        ws << L"Wyjątek OpenVINO podczas optymalizacji modelu: " << std::wstring(e.what(), e.what() + strlen(e.what())) << L"\n";
        updateErrorEditText(hwndErrorEdit, ws.str());
    }
    catch (const std::exception& e) {
        std::wstringstream ws;
        ws << L"Standardowy wyjątek podczas optymalizacji modelu: " << std::wstring(e.what(), e.what() + strlen(e.what())) << L"\n";
        updateErrorEditText(hwndErrorEdit, ws.str());
    }
    catch (...) {
        updateErrorEditText(hwndErrorEdit, L"Nieznany wyjątek podczas optymalizacji modelu\n");
    }
}



void matrix_multiply_avx2(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    if (A.size() < N * N || B.size() < N * N || C.size() < N * N) {
        std::cerr << "Rozmiar wektorów A, B lub C jest nieprawidłowy.\n";
        return;
    }

    auto is_aligned = [](const void* ptr) {
        return reinterpret_cast<uintptr_t>(ptr) % 32 == 0;
        };

    if (!is_aligned(A.data()) || !is_aligned(B.data()) || !is_aligned(C.data())) {
        std::cerr << "Dane nie są wyrównane do 32 bajtów!\n";
        return;
    }

    for (int i = 0; i < N; i += 8) {
        for (int j = 0; j < N; j++) {
            __m256 c = _mm256_setzero_ps();
            for (int k = 0; k < N; k++) {
                __m256 a = _mm256_load_ps(&A[i * N + k * 8]);
                __m256 b = _mm256_set1_ps(B[k * N + j]);
                c = _mm256_fmadd_ps(a, b, c);
            }
            _mm256_store_ps(&C[i * N + j * 8], c);
        }
    }

    // Obsługa przypadków, gdy N nie jest wielokrotnością 8
    int remainder = N % 8;
    if (remainder != 0) {
        for (int i = N - remainder; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float c = 0.0f;
                for (int k = 0; k < N; ++k) {
                    c += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = c;
            }
        }
    }
}



void convert_to_openvino_ir(const DeepNeuralNetwork& nn, const std::string& xml_path, const std::string& bin_path) {
    // Tworzenie modelu OpenVINO
    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 1, static_cast<size_t>(nn.get_weights()[0].cols()) });
    std::shared_ptr<ov::Node> current = input;

    // Dodawanie warstw do modelu
    for (size_t i = 0; i < nn.get_weights().size(); ++i) {
        auto weights = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape{ static_cast<size_t>(nn.get_weights()[i].rows()), static_cast<size_t>(nn.get_weights()[i].cols()) }, nn.get_weights()[i].data());
        auto biases = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape{ 1, static_cast<size_t>(nn.get_weights()[i].rows()) }, nn.get_biases()[i].data());

        // Sprawdzenie zgodności wymiarów do mnożenia macierzy
        if (current->get_shape().back() != weights->get_shape().back()) {
            throw std::runtime_error("Niezgodność wymiarów: Nie można mnożyć macierzy o niezgodnych wymiarach.");
        }

        // Debug: Wypisanie wymiarów macierzy
        std::wstringstream ws;
        ws << L"Warstwa " << i << L":" << std::endl;
        ws << L"  Aktualny kształt: " << ov::util::to_string(current->get_shape()).c_str() << std::endl;
        ws << L"  Kształt wag: " << ov::util::to_string(weights->get_shape()).c_str() << std::endl;
        LogToEditControl(hwndErrorEdit, ws.str());

        auto matmul = std::make_shared<ov::opset8::MatMul>(current, weights, false, true);
        auto add = std::make_shared<ov::opset8::Add>(matmul, biases);
        current = std::make_shared<ov::opset8::Relu>(add); // Zakładamy aktywację ReLU dla uproszczenia
    }

    // Tworzenie funkcji OpenVINO
    auto result = std::make_shared<ov::opset8::Result>(current);
    auto function = std::make_shared<ov::Model>(ov::ResultVector{ result }, ov::ParameterVector{ input });

    // Serializacja modelu do plików XML i BIN
    ov::serialize(function, xml_path, bin_path);
}




void train_with_npu(std::shared_ptr<RecurrentNeuralNetwork> rnn, const vector<VectorXd>& inputs, const vector<VectorXd>& targets, double learning_rate, int epochs, HWND hwnd) {
    if (inputs.size() != targets.size()) {
        LogErrorToEditControl(hwndErrorEdit, L"Rozmiary inputs i targets są niezgodne.\n");
        return;
    }

    std::thread npu_thread([=]() {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<std::future<void>> futures;
            for (size_t i = 0; i < inputs.size(); ++i) {
                futures.push_back(std::async(std::launch::async, [&, i]() {
                    try {
                        std::lock_guard<std::mutex> lock(model_mutex); // Dodaj blokadę mutexu

                        if (inputs[i].size() != rnn->get_input_size()) {
                            LogErrorToEditControl(hwndErrorEdit, L"Rozmiar wejścia jest niezgodny z oczekiwanym rozmiarem sieci.\n");
                            return;
                        }

                        if (targets[i].size() != rnn->get_output_size()) {
                            LogErrorToEditControl(hwndErrorEdit, L"Rozmiar celu jest niezgodny z oczekiwanym rozmiarem wyjścia sieci.\n");
                            return;
                        }

                        // Przenieś obliczenia na NPU
                        std::vector<float> input_data(inputs[i].data(), inputs[i].data() + inputs[i].size());
                        std::vector<float> output_data;
                        run_cnn_inference(rnn->get_model(), input_data, output_data);

                        // Poprawione wywołanie Eigen::Map
                        Eigen::Map<Eigen::VectorXf> output(output_data.data(), output_data.size());
                        VectorXd reward = targets[i] - output.cast<double>();

                        if (reward.size() != rnn->get_output_size()) {
                            LogErrorToEditControl(hwndErrorEdit, L"Rozmiar reward jest niezgodny z oczekiwanym rozmiarem wyjścia sieci.\n");
                            return;
                        }

                        rnn->train({ inputs[i] }, { reward }, learning_rate);

                        double loss = reward.squaredNorm();
                        double accuracy = (reward.array().abs() < 0.5).cast<double>().mean();
                        AddDataPoint(loss, accuracy);
                    }
                    catch (const std::invalid_argument& e) {
                        std::wstringstream ws;
                        ws << L"Wyjątek std::invalid_argument: " << e.what() << std::endl;
                        LogErrorToEditControl(hwndErrorEdit, ws.str());
                    }
                    catch (const std::exception& e) {
                        std::wstringstream ws;
                        ws << L"Standardowy wyjątek: " << e.what() << std::endl;
                        LogErrorToEditControl(hwndErrorEdit, ws.str());
                    }
                    catch (...) {
                        LogErrorToEditControl(hwndErrorEdit, L"Nieznany wyjątek.\n");
                    }
                    }));
            }

            for (auto& future : futures) {
                future.get();
            }

            std::wstringstream ws;
            ws << L"Epoch " << epoch + 1 << L" completed.\n";
            LogToEditControl(hwndErrorEdit, ws.str());

            {
                std::lock_guard<std::mutex> lock(mtx);
                if (IsWindow(hwnd)) {
                    PostMessage(hwnd, WM_USER + 1, static_cast<WPARAM>(epoch + 1), 0);
                }
            }

            {
                std::lock_guard<std::mutex> lock(cv_mtx);
                if (!training_in_progress) {
                    LogToEditControl(hwndErrorEdit, L"Trening przerwany.\n");
                    return;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            if (IsWindow(hwnd)) {
                PostMessage(hwnd, WM_USER + 2, 0, 0);
            }
        }
        });

    npu_thread.detach();
}


void LoadAndDisplaylangAnalyzerCResults(HWND hwndErrorEdit) {
    std::ifstream file("clang_analyzer_results.txt");
    if (!file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku clang_analyzer_results.txt.\n");
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string results = buffer.str();
    file.close();

    std::wstring wresults;
    std::istringstream iss(results);
    std::string line;
    std::regex error_regex(R"((.*):(\d+):(\d+): (warning|error|note): (.*))");
    std::smatch match;

    CHARFORMAT2 cf;
    cf.cbSize = sizeof(CHARFORMAT2);
    cf.dwMask = CFM_COLOR;
    cf.dwEffects = 0;

    while (std::getline(iss, line)) {
        if (std::regex_match(line, match, error_regex)) {
            std::string file = match[1];
            std::string line_number = match[2];
            std::string column_number = match[3];
            std::string type = match[4];
            std::string message = match[5];

            std::wstring wline = std::wstring(line.begin(), line.end());
            std::wstring wfile = std::wstring(file.begin(), file.end());
            std::wstring wline_number = std::wstring(line_number.begin(), line_number.end());
            std::wstring wcolumn_number = std::wstring(column_number.begin(), column_number.end());
            std::wstring wtype = std::wstring(type.begin(), type.end());
            std::wstring wmessage = std::wstring(message.begin(), message.end());

            if (type == "warning") {
                cf.crTextColor = RGB(255, 165, 0);
                SendMessage(hwndErrorEdit, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
                wresults += L"⚠️ Ostrzeżenie: " + wfile + L":" + wline_number + L":" + wcolumn_number + L" - " + wmessage + L"\n";
            }
            else if (type == "error") {
                cf.crTextColor = RGB(255, 0, 0);
                SendMessage(hwndErrorEdit, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
                wresults += L"❌ Błąd: " + wfile + L":" + wline_number + L":" + wcolumn_number + L" - " + wmessage + L"\n";
            }
            else if (type == "note") {
                cf.crTextColor = RGB(0, 0, 255);
                SendMessage(hwndErrorEdit, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
                wresults += L"ℹ️ Notatka: " + wfile + L":" + wline_number + L":" + wcolumn_number + L" - " + wmessage + L"\n";
            }
        }
        else {
            wresults += std::wstring(line.begin(), line.end()) + L"\n";
        }
    }

    SetWindowTextW(hwndErrorEdit, wresults.c_str());
}


void runClangStaticAnalyzer() {
    std::lock_guard<std::mutex> lock(clang_mutex); // Dodaj blokadę mutexu

    std::string command = "clang --analyze -Xanalyzer -analyzer-output=text "
        "-Xanalyzer -analyzer-checker=core "
        "-Xanalyzer -analyzer-checker=deadcode "
        "-Xanalyzer -analyzer-checker=security "
        "-Xanalyzer -analyzer-checker=optin.cplusplus.VirtualCall "
        "-Xanalyzer -analyzer-checker=core.NullDereference "
        "-Xanalyzer -analyzer-checker=core.DivideZero "
        "-Xanalyzer -analyzer-checker=core.StackAddressEscape "
        "-Xanalyzer -analyzer-checker=core.UndefinedBinaryOperatorResult "
        "-Xanalyzer -analyzer-checker=core.CallAndMessage "
        "-Xanalyzer -analyzer-checker=core.NonNullParamChecker "
        "-Xanalyzer -analyzer-checker=core.builtin.NoReturnFunctions "
        "-Xanalyzer -analyzer-checker=core.builtin.BuiltinFunctions "
        "-I C:/Users/MOREK/source/repos/WindowsProject1/vcpkg/installed/x64-windows/include/eigen3 "
        "-I C:/Users/MOREK/source/repos/WindowsProject1/vcpkg/installed/x64-windows/include "
        "C:/Users/MOREK/source/repos/WindowsProject1/WindowsProject1/*.cpp -v 2> clang_analyzer_results.txt";
    std::system(command.c_str());
}




// Inicjalizacja RichEdit
void InitializeRichEdit() {
    LoadLibrary(TEXT("Msftedit.dll"));
}
void LoadAndDisplayClangAnalyzerResults(HWND hwndErrorEdit) {
    std::ifstream file("clang_analyzer_results.txt");
    if (!file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku clang_analyzer_results.txt.\n");
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string results = buffer.str();
    file.close();

    std::wstring wresults;
    std::istringstream iss(results);
    std::string line;
    std::regex error_regex(R"((.*):(\d+):(\d+): (warning|error|note): (.*))");
    std::smatch match;

    CHARFORMAT2 cf;
    cf.cbSize = sizeof(CHARFORMAT2);
    cf.dwMask = CFM_COLOR;
    cf.dwEffects = 0;

    while (std::getline(iss, line)) {
        if (std::regex_match(line, match, error_regex)) {
            std::string file = match[1];
            std::string line_number = match[2];
            std::string column_number = match[3];
            std::string type = match[4];
            std::string message = match[5];

            std::wstring wline = std::wstring(line.begin(), line.end());
            std::wstring wfile = std::wstring(file.begin(), file.end());
            std::wstring wline_number = std::wstring(line_number.begin(), line_number.end());
            std::wstring wcolumn_number = std::wstring(column_number.begin(), column_number.end());
            std::wstring wtype = std::wstring(type.begin(), type.end());
            std::wstring wmessage = std::wstring(message.begin(), message.end());

            if (type == "warning") {
                cf.crTextColor = RGB(255, 165, 0);
                SendMessage(hwndErrorEdit, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
                wresults += L"⚠️ Ostrzeżenie: " + wfile + L":" + wline_number + L":" + wcolumn_number + L" - " + wmessage + L"\n";
            }
            else if (type == "error") {
                cf.crTextColor = RGB(255, 0, 0);
                SendMessage(hwndErrorEdit, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
                wresults += L"❌ Błąd: " + wfile + L":" + wline_number + L":" + wcolumn_number + L" - " + wmessage + L"\n";
            }
            else if (type == "note") {
                cf.crTextColor = RGB(0, 0, 255);
                SendMessage(hwndErrorEdit, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
                wresults += L"ℹ️ Notatka: " + wfile + L":" + wline_number + L":" + wcolumn_number + L" - " + wmessage + L"\n";
            }
        }
        else {
            wresults += std::wstring(line.begin(), line.end()) + L"\n";
        }
    }

    SetWindowTextW(hwndErrorEdit, wresults.c_str());
}



// Funkcja do aktualizacji kontrolki hwndEdit
void updateEditText(HWND hwndEdit, const std::string& text) {
    std::wstring wtext = StringToWideString(text);
    {
        std::lock_guard<std::mutex> lock(editTextMutex);
        SetWindowTextW(hwndEdit, wtext.c_str());
    }
}

void someFunctionThatModifiesCode() {
    std::string modifiedCode = "Nowy kod źródłowy";

    {
        std::lock_guard<std::mutex> lock(codeMutex);
        kod_zrodlowy = modifiedCode;
        kod_zmieniony = true;
    }

    // Aktualizacja kontrolki hwndEdit
    updateEditText(hwndEdit, modifiedCode);
}

void monitorCodeChanges() {
    std::string poprzedni_kod;
    while (true) {
        bool zmieniony = false;
        {
            std::lock_guard<std::mutex> lock(codeMutex);
            if (kod_zrodlowy != poprzedni_kod) {
                kod_zmieniony = true;
                poprzedni_kod = kod_zrodlowy;
                zmieniony = true;
            }
        }
        if (zmieniony) {
            kod_zmieniony = false;
            updateEditText(hwndEdit, poprzedni_kod);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sprawdź co 100 ms
    }
}

void initializeMonitoring() {
    std::thread monitorThread(monitorCodeChanges);
    monitorThread.detach();
}







// Struktura przechowująca propozycję zmiany
// Struktura przechowująca propozycję zmiany
struct ChangeProposal {
    std::string original_code;
    std::string modified_code;
    std::unique_ptr<double> confidence_level;
    std::string description;

    // Konstruktor domyślny inicjalizujący zmienne składowe
    ChangeProposal()
        : original_code(""), modified_code(""), confidence_level(std::make_unique<double>(0.0)), description("") {
    }

    // Konstruktor przyjmujący argumenty do inicjalizacji zmiennych składowych
    ChangeProposal(const std::string& original, const std::string& modified, double confidence, const std::string& desc)
        : original_code(original), modified_code(modified), confidence_level(std::make_unique<double>(confidence)), description(desc) {
    }

    // Konstruktor kopiujący
    ChangeProposal(const ChangeProposal& other)
        : original_code(other.original_code), modified_code(other.modified_code), confidence_level(std::make_unique<double>(*other.confidence_level)), description(other.description) {
    }

    // Operator przypisania kopiującego
    ChangeProposal& operator=(const ChangeProposal& other) {
        if (this == &other) return *this;
        original_code = other.original_code;
        modified_code = other.modified_code;
        confidence_level = std::make_unique<double>(*other.confidence_level);
        description = other.description;
        return *this;
    }

    // Konstruktor przenoszący
    ChangeProposal(ChangeProposal&& other) noexcept
        : original_code(std::move(other.original_code)), modified_code(std::move(other.modified_code)), confidence_level(std::move(other.confidence_level)), description(std::move(other.description)) {
    }

    // Operator przypisania przenoszącego
    ChangeProposal& operator=(ChangeProposal&& other) noexcept {
        if (this == &other) return *this;
        original_code = std::move(other.original_code);
        modified_code = std::move(other.modified_code);
        confidence_level = std::move(other.confidence_level);
        description = std::move(other.description);
        return *this;
    }
};


// Funkcja uruchamiająca testy jednostkowe
bool runUnitTests() {
    std::string command = "ctest --output-on-failure";
    int result = std::system(command.c_str());
    return result == 0;
}

// Funkcja uruchamiająca analizę statyczną za pomocą clang-tidy
bool runStaticAnalysis(const std::string& code) {
    std::ofstream file("temp_code.cpp");
    if (!file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku do zapisu kodu.\n");
        return false;
    }
    file << code;
    file.close();

    std::string command = "clang-tidy temp_code.cpp -- -std=c++17";
    int result = std::system(command.c_str());
    std::remove("temp_code.cpp");
    return result == 0;
}



// Funkcja do obliczania poziomu pewności na podstawie wyników testów i analiz
double calculateConfidenceLevel(const std::string& modified_code) {
    double confidence = 0.0;
    int total_checks = 0;

    // Analiza statyczna
    if (runStaticAnalysis(modified_code)) {
        confidence += 20.0; // Przykładowa wartość
    }
    total_checks += 20;

    // Testy jednostkowe
    if (runUnitTests()) {
        confidence += 30.0; // Przykładowa wartość
    }
    total_checks += 30;

    // Testy logiczne
    double logical_test_coverage = run_logical_tests(modified_code);
    confidence += logical_test_coverage * 50.0; // Przykładowa wartość
    total_checks += 50;

    // Obliczanie końcowego poziomu pewności
    return (confidence / total_checks) * 100.0;
}

// Funkcja generująca propozycję zmiany
ChangeProposal generateChangeProposal(const std::string& original_code) {
    ChangeProposal proposal;
    proposal.original_code = original_code;
    proposal.modified_code = "for (int index = 0; index < 10; ++index)"; // Przykładowa zmiana
    *proposal.confidence_level = calculateConfidenceLevel(proposal.modified_code); // Dynamiczne obliczanie poziomu pewności
    proposal.description = "Zmień nazwę zmiennej 'i' na 'index' w pętli.";
    return proposal;
}




// Funkcja oceniająca propozycję zmiany
bool evaluateChangeProposal(const ChangeProposal& proposal) {
    // Przykładowe kryterium: akceptuj zmiany z pewnością powyżej 90%
    if (*proposal.confidence_level <= 90.0) {
        return false;
    }

    // Analiza statyczna
    if (!runStaticAnalysis(proposal.modified_code)) {
        LogErrorToEditControl(hwndErrorEdit, L"Analiza statyczna nie powiodła się.\n");
        return false;
    }

    // Sprawdzanie kontekstu
    // Przykładowa logika sprawdzania kontekstu
    if (proposal.modified_code.find("int main") != std::string::npos) {
        LogErrorToEditControl(hwndErrorEdit, L"Zmiana w funkcji main nie jest dozwolona.\n");
        return false;
    }

    // Uruchamianie testów jednostkowych
    if (!runUnitTests()) {
        LogErrorToEditControl(hwndErrorEdit, L"Testy jednostkowe nie powiodły się.\n");
        return false;
    }

    // Definiowane reguły
    std::regex comment_regex(R"(//.*)");
    if (!std::regex_search(proposal.modified_code, comment_regex)) {
        LogErrorToEditControl(hwndErrorEdit, L"Zmiana musi zawierać komentarze.\n");
        return false;
    }

    return true;
}

// Funkcja stosująca zmiany do kodu
void applyChangeProposal(const ChangeProposal& proposal) {
    std::lock_guard<std::mutex> lock(codeMutex);
    kod_zrodlowy = proposal.modified_code;
    kod_zmieniony = true;
}

// Funkcja logująca zmiany
void logChange(const ChangeProposal& proposal) {
    std::ofstream log_file("change_log.txt", std::ios::app);
    if (log_file.is_open()) {
        log_file << "Oryginalny kod:\n" << proposal.original_code << "\n";
        log_file << "Zmodyfikowany kod:\n" << proposal.modified_code << "\n";
        log_file << "Pewność modelu: " << *proposal.confidence_level << "%\n";
        log_file << "Opis zmiany: " << proposal.description << "\n";
        log_file << "--------------------------\n";
        log_file.close();
    }
}

// Funkcja przetwarzająca propozycje zmian
void processChangeProposal(const ChangeProposal& proposal) {
    if (evaluateChangeProposal(proposal)) {
        applyChangeProposal(proposal);
        logChange(proposal);
        updateEditText(hwndEdit, proposal.modified_code);
        // Informowanie użytkownika o automatycznych zmianach
        LogToEditControl(hwndErrorEdit, L"Automatyczna zmiana została wprowadzona.\n");
    }
}


// Funkcja wyświetlająca propozycję zmiany w kontrolce hwndEdit
void displayChangeProposal(HWND hwndEdit, const ChangeProposal& proposal) {
    std::wstringstream ws;
    ws << L"Oryginalny kod:\n" << StringToWideString(proposal.original_code) << L"\n\n";
    ws << L"Zmodyfikowany kod:\n" << StringToWideString(proposal.modified_code) << L"\n\n";
    ws << L"Pewność modelu: " << std::fixed << std::setprecision(2) << *proposal.confidence_level << L"%\n\n";
    ws << L"Opis zmiany: " << StringToWideString(proposal.description) << L"\n";

    SetWindowTextW(hwndEdit, ws.str().c_str());
}


// Globalne zmienne do obsługi kolejki komunikatów
std::queue<std::shared_ptr<ChangeProposal>> messageQueue;
std::mutex queueMutex;
std::condition_variable queueCondition;

// Funkcja dodająca komunikat do kolejki
void add_message(const std::shared_ptr<ChangeProposal>& message) {
    std::lock_guard<std::mutex> lock(queueMutex);
    messageQueue.push(message);
    queueCondition.notify_one();
}

// Funkcja pobierająca komunikat z kolejki
std::shared_ptr<ChangeProposal> get_change_proposal_message() {
    std::unique_lock<std::mutex> lock(queueMutex);
    queueCondition.wait(lock, [] { return !messageQueue.empty(); });
    auto message = messageQueue.front();
    messageQueue.pop();
    return message;
}

// Funkcja przetwarzająca komunikaty z kolejki
void messageProcessingThread(HWND hwnd) {
    while (running) {
        auto message = get_change_proposal_message();
        // Przetwarzanie komunikatu
        if (message->description.find("ChangeProposal") != std::string::npos) {
            // Deserializacja propozycji zmiany
            ChangeProposal proposal = *message;
            // Kod deserializacji tutaj (np. z JSON)
            processChangeProposal(proposal);

            // Aktualizacja kontrolki hwndEdit
            PostMessage(hwnd, WM_UPDATE_EDIT, 0, (LPARAM)new std::string(proposal.modified_code));

            // Aktualizacja kontrolki hwndErrorEdit
            PostMessage(hwnd, WM_UPDATE_ERROR_EDIT, 0, (LPARAM)new std::wstring(L"Automatyczna zmiana została wprowadzona.\n"));
        }
        else if (message->description == "MessageTypeTHREAD_TERMINATE") {
            break;
        }
        // Inne typy komunikatów można obsłużyć tutaj
    }
}



// Przykład użycia
void exampleUsage(HWND hwndEdit) {
    std::string original_code = "for (int i = 0; i < 10; ++i)";
    auto proposal = std::make_shared<ChangeProposal>(generateChangeProposal(original_code));
    displayChangeProposal(hwndEdit, *proposal);

    // Przekazywanie propozycji zmiany do kolejki komunikatów
    add_message(proposal);
}

// Historia zmian
std::stack<ChangeProposal> change_history;

// Funkcja cofająca ostatnie zmiany
void undoLastChanges(int num_changes) {
    for (int i = 0; i < num_changes && !change_history.empty(); ++i) {
        ChangeProposal last_change = change_history.top();
        change_history.pop();
        applyChangeProposal({ last_change.modified_code, last_change.original_code, 100.0, "Cofnięcie zmiany" });
        updateEditText(hwndEdit, last_change.original_code);
        LogToEditControl(hwndErrorEdit, L"Automatyczna zmiana została cofnięta.\n");
    }
}

// Przykład wywołania funkcji undoLastChanges w odpowiednim miejscu
void someFunction() {
    // Cofnięcie do 10 ostatnich zmian
    undoLastChanges(10);
}




// Konfiguracja kryteriów oceny
double confidence_threshold = 90.0;

void setConfidenceThreshold(double threshold) {
    confidence_threshold = threshold;
}

enum MessageType {
    // Inne typy komunikatów
    MessageTypeTHREAD_TERMINATE
};

class Singleton {
public:
    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }

    void Inicjalizacja() {
        if (!initialized) {
            // Wczytywanie konfiguracji
            // Inicjalizacja modelu RNN
            rnn = std::make_shared<RecurrentNeuralNetwork>(input_size, hidden_size, output_size);
            rnn->load_model("rnn_model.bin");
            initialized = true;
        }
    }

    std::shared_ptr<RecurrentNeuralNetwork> getRNN() {
        return rnn;
    }

private:
    Singleton() : initialized(false) {}
    bool initialized;
    std::shared_ptr<RecurrentNeuralNetwork> rnn;
    int input_size = 224 * 224 * 3;
    int hidden_size = 128;
    int output_size = 10;
};



LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hwndEdit;
    static HWND hwndErrorEdit;
    static int current_epoch = 0;
    static std::shared_ptr<RecurrentNeuralNetwork> rnn;
    static std::shared_ptr<ov::Model> cnn_model;
    static ov::Core core;

    switch (uMsg) {
    case WM_CREATE:
    {
        InitializeRichEdit();

        hwndEdit = CreateWindowExW(
            WS_EX_CLIENTEDGE,
            MSFTEDIT_CLASS,
            NULL,
            WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL,
            61, 54, 400, 200,
            hwnd,
            (HMENU)ID_EDIT_TEXT,
            GetModuleHandle(NULL),
            NULL
        );

        hwndErrorEdit = CreateWindowExW(
            WS_EX_CLIENTEDGE,
            MSFTEDIT_CLASS,
            NULL,
            WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL,
            10, 270, 400, 200,
            hwnd,
            NULL,
            GetModuleHandle(NULL),
            NULL
        );

        runClangStaticAnalyzer();
        LoadAndDisplayClangAnalyzerResults(hwndErrorEdit);

        // Inicjalizacja modelu CNN
        load_cnn_model("model.xml", "model.bin", core, cnn_model);

        // Inicjalizacja Singleton
        Singleton::getInstance().Inicjalizacja();
        rnn = Singleton::getInstance().getRNN();

        // Inicjalizacja monitorowania
        initializeMonitoring();

        // Wywołanie funkcji generującej propozycję zmiany
        exampleUsage(hwndEdit);

        // Uruchomienie wątku przetwarzającego komunikaty
        std::thread messageThread(messageProcessingThread, hwnd);
        messageThread.detach();

        return 0;
    }

    case WM_SIZE:
    {
        RECT rect;
        GetClientRect(hwnd, &rect);
        SetWindowPos(hwndErrorEdit, NULL, rect.right - 410, rect.bottom - 230, 400, 200, SWP_NOZORDER);
        InvalidateRect(hwnd, NULL, TRUE);
        return 0;
    }

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        std::wstring status_text;
        std::wstring epoch_text;
        {
            std::lock_guard<std::mutex> lock(mtx);
            status_text = training_in_progress ? L"Trening w toku..." : L"Trening zakończony";
            epoch_text = L"Epoka: " + std::to_wstring(current_epoch);
        }
        TextOutW(hdc, 5, 5, status_text.c_str(), status_text.length());
        TextOutW(hdc, 5, 25, epoch_text.c_str(), epoch_text.length());

        RECT client_rect;
        GetClientRect(hwnd, &client_rect);

        RECT loss_rect = { client_rect.right - 410, 10, client_rect.right - 10, 160 };
        RECT accuracy_rect = { client_rect.right - 410, 170, client_rect.right - 10, 320 };
        DrawGraph(hdc, loss_rect, loss_values, L"Loss");
        DrawGraph(hdc, accuracy_rect, accuracy_values, L"Accuracy");

        RECT diagnostic_rect = { 712, 336, client_rect.right - 10, client_rect.bottom - 10 };
        std::wstring auto_percentage = std::to_wstring((auto_data_count * 100) / total_data_count) + L"%";
        std::wstring manual_percentage = std::to_wstring((manual_data_count * 100) / total_data_count) + L"%";
        std::wstring diagnostic_text = L"Wczytano dane treningowe: automatyczne " + auto_percentage + L", ręczne " + manual_percentage;
        DrawDiagnosticText(hdc, diagnostic_rect, diagnostic_text);

        RECT error_label_rect = { client_rect.right - 410, client_rect.bottom - 250, client_rect.right - 10, client_rect.bottom - 230 };
        DrawTextW(hdc, L"Błędy logiczne i inne błędy", -1, &error_label_rect, DT_CENTER | DT_TOP | DT_SINGLELINE);

        EndPaint(hwnd, &ps);
        return 0;
    }

    case WM_USER + 1:
    {
        std::lock_guard<std::mutex> lock(mtx);
        current_epoch = static_cast<int>(wParam);
    }
    InvalidateRect(hwnd, NULL, TRUE);
    return 0;
    case WM_USER + 2:
    {
        std::lock_guard<std::mutex> lock(mtx);
        training_in_progress = false;
    }
    InvalidateRect(hwnd, NULL, TRUE);
    return 0;
    case WM_UPDATE_EDIT:
    {
        std::string* modified_code = reinterpret_cast<std::string*>(lParam);
        updateEditText(hwndEdit, *modified_code);
        delete modified_code; // Zwalnianie pamięci
    }
    return 0;
    case WM_UPDATE_ERROR_EDIT:
    {
        std::wstring* message = reinterpret_cast<std::wstring*>(lParam);
        LogToEditControl(hwndErrorEdit, *message);
        delete message; // Zwalnianie pamięci
    }
    return 0;

    case WM_CLOSE:
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        running = false;
        message_queue.push("MessageTypeTHREAD_TERMINATE");
        queue_cv.notify_one();
        if (messageThread.joinable()) {
            messageThread.join();
        }
        DestroyWindow(hwnd);
    }
    return 0;
    case WM_KEYDOWN:
    {
        if (wParam == VK_ESCAPE) {
            DestroyWindow(hwnd);
        }
    }
    return 0;
    case WM_LBUTTONDOWN:
    {
        POINTS pt = MAKEPOINTS(lParam);
        std::wstringstream ws;
        ws << L"Kliknięto w punkcie: (" << pt.x << L", " << pt.y << L")";
        LogToEditControl(hwndErrorEdit, ws.str() + L"\n");
    }
    return 0;
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}


void updateUIAfterModification(HWND hwndEdit, const std::string& modified_code) {
    updateEditText(hwndEdit, modified_code);
    InvalidateRect(GetParent(hwndEdit), NULL, TRUE);

    // Wywołanie funkcji run_logical_tests
    if (run_logical_tests(modified_code)) {
        LogToEditControl(hwndErrorEdit, L"Testy logiczne przeszły pomyślnie.\n");
    }
    else {
        LogErrorToEditControl(hwndErrorEdit, L"Testy logiczne nie powiodły się.\n");
    }
}



VectorXd reprezentacja_wektorowa(const string& kod_zrodlowy) {
    std::lock_guard<std::mutex> lock(clang_mutex);

    CXIndex index = clang_createIndex(0, 0);
    unsigned int options = CXTranslationUnit_None;
    CXUnsavedFile unsavedFile = { "temp_code.cpp", kod_zrodlowy.c_str(), static_cast<unsigned long>(kod_zrodlowy.size()) };
    CXTranslationUnit unit;
    if (clang_parseTranslationUnit2(index, "temp_code.cpp", nullptr, 0, &unsavedFile, 1, options, &unit) != CXError_Success) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można sparsować kodu źródłowego.\n");
        clang_disposeIndex(index);
        return VectorXd::Zero(224 * 224 * 3);
    }
    CXCursor cursor = clang_getTranslationUnitCursor(unit);
    VectorXd features = VectorXd::Zero(224 * 224 * 3);
    int feature_index = 0;
    auto data = std::make_pair(&features, &feature_index);
    clang_visitChildren(
        cursor,
        [](CXCursor c, CXCursor parent, CXClientData client_data) {
            auto data = static_cast<std::pair<VectorXd*, int*>*>(client_data);
            VectorXd* features = data->first;
            int* feature_index = data->second;
            if (clang_Location_isFromMainFile(clang_getCursorLocation(c)) == 0) {
                return CXChildVisit_Continue;
            }
            CXCursorKind kind = clang_getCursorKind(c);
            if (kind == CXCursor_FunctionDecl) {
                if (*feature_index < features->size()) {
                    (*features)(*feature_index) = 1.0; // Oznacz funkcję
                    (*feature_index)++;
                }
            }
            else if (kind == CXCursor_ClassDecl) {
                if (*feature_index < features->size()) {
                    (*features)(*feature_index) = 2.0; // Oznacz klasę
                    (*feature_index)++;
                }
            }
            else if (kind == CXCursor_VarDecl) {
                if (*feature_index < features->size()) {
                    (*features)(*feature_index) = 3.0; // Oznacz zmienną
                    (*feature_index)++;
                }
            }
            else if (kind == CXCursor_ParmDecl) {
                if (*feature_index < features->size()) {
                    (*features)(*feature_index) = 4.0; // Oznacz parametr funkcji
                    (*feature_index)++;
                }
            }
            // Dodanie dodatkowych cech
            else if (kind == CXCursor_CallExpr) {
                if (*feature_index < features->size()) {
                    (*features)(*feature_index) = 5.0; // Oznacz wywołanie funkcji
                    (*feature_index)++;
                }
            }
            else if (kind == CXCursor_ReturnStmt) {
                if (*feature_index < features->size()) {
                    (*features)(*feature_index) = 6.0; // Oznacz instrukcję return
                    (*feature_index)++;
                }
            }
            return CXChildVisit_Recurse;
        },
        &data);
    clang_disposeTranslationUnit(unit);
    clang_disposeIndex(index);
    return features;
}

// Funkcja generująca modyfikacje na podstawie wyników modelu
std::string przekształć_na_kod_zrodlowy(const VectorXd& output, int attempts) {
    std::stringstream ss;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);
    std::uniform_int_distribution<> dist_type(0, 14);

    for (int i = 0; i < output.size(); ++i) {
        double value = output[i];

        if (attempts > 0) {
            value += dis(gen) * attempts;
        }

        int mod_type = dist_type(gen);

        if (attempts % 3 == 0) {
            if (value > 0.5) {
                if (mod_type == 0) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    int c = a + b;\n    std::cout << \"Wynik: \" << c << std::endl;\n}\n";
                }
                else if (mod_type == 1) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    for (int j = 0; j < a; ++j) {\n        std::cout << j << std::endl;\n    }\n}\n";
                }
                else if (mod_type == 2) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    if (a > b) {\n        std::cout << \"a > b\" << std::endl;\n    } else {\n        std::cout << \"a <= b\" << std::endl;\n    }\n}\n";
                }
                else if (mod_type == 3) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    int arr[5] = {1, 2, 3, 4, 5};\n    for (int j = 0; j < 5; ++j) {\n        std::cout << arr[j] << std::endl;\n    }\n}\n";
                }
                else if (mod_type == 4) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    std::vector<int> vec = {1, 2, 3, 4, 5};\n    for (int val : vec) {\n        std::cout << val << std::endl;\n    }\n}\n";
                }
                else if (mod_type == 5) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    std::map<int, std::string> mapa = {{1, \"jeden\"}, {2, \"dwa\"}};\n    for (const auto& [klucz, wartosc] : mapa) {\n        std::cout << klucz << \": \" << wartosc << std::endl;\n    }\n}\n";
                }
                else if (mod_type == 6) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    std::set<int> zbior = {1, 2, 3, 4, 5};\n    for (int val : zbior) {\n        std::cout << val << std::endl;\n    }\n}\n";
                }
                else if (mod_type == 7) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    std::list<int> lista = {1, 2, 3, 4, 5};\n    for (int val : lista) {\n        std::cout << val << std::endl;\n    }\n}\n";
                }
                else if (mod_type == 8) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    std::deque<int> kolejka = {1, 2, 3, 4, 5};\n    for (int val : kolejka) {\n        std::cout << val << std::endl;\n    }\n}\n";
                }
                else if (mod_type == 9) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    std::array<int, 5> tablica = {1, 2, 3, 4, 5};\n    for (int val : tablica) {\n        std::cout << val << std::endl;\n    }\n}\n";
                }
                else if (mod_type == 10) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    int zmienna_" << i << " = a + b;\n    std::cout << \"Zmienna: \" << zmienna_" << i << " << std::endl;\n}\n";
                }
                else if (mod_type == 11) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    double zmienna_" << i << " = a * b;\n    std::cout << \"Zmienna: \" << zmienna_" << i << " << std::endl;\n}\n";
                }
                else if (mod_type == 12) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    std::string zmienna_" << i << " = std::to_string(a) + std::to_string(b);\n    std::cout << \"Zmienna: \" << zmienna_" << i << " << std::endl;\n}\n";
                }
                else if (mod_type == 13) {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    // To jest komentarz\n    std::cout << \"Komentarz\" << std::endl;\n}\n";
                }
                else {
                    ss << "void funkcja_" << i << "(int a, int b) {\n    /* To jest komentarz blokowy */\n    std::cout << \"Komentarz blokowy\" << std::endl;\n}\n";
                }
            }
            else if (value > 0.3) {
                ss << "void zmien_parametry_funkcji_" << i << "(int a, int b) {\n    int wynik = a * b;\n    std::cout << \"Wynik: \" << wynik << std::endl;\n}\n";
            }
            else if (value > 0.1) {
                ss << "class Klasa_" << i << " {\npublic:\n    void metoda(int a, int b) {\n        int x = a + b;\n        int y = a * b;\n        int z = x * y;\n        std::cout << \"Wynik: \" << z << std::endl;\n    }\n};\n";
            }
        }
        else if (attempts % 3 == 1) {
            if (value > 0.5) {
                ss << "void funkcja_" << i << "(int a, int b) {\n    int x = a + b;\n    int y = a - b;\n    int z = x * y;\n    std::cout << \"Wynik: \" << z << std::endl;\n}\n";
            }
            else if (value > 0.3) {
                ss << "void zmien_parametry_funkcji_" << i << "(int a, int b) {\n    int wynik = a + b;\n    std::cout << \"Wynik: \" << wynik << std::endl;\n}\n";
            }
        }
        else {
            if (value > 0.5) {
                ss << "void funkcja_" << i << "(int a, int b) {\n    int c = a * b;\n    std::cout << \"Wynik: \" << c << std::endl;\n}\n";
            }
            else if (value > 0.3) {
                ss << "void zmien_parametry_funkcji_" << i << "(int a, int b) {\n    int wynik = a - b;\n    std::cout << \"Wynik: \" << wynik << std::endl;\n}\n";
            }
            else if (value > 0.1) {
                ss << "class Klasa_" << i << " {\npublic:\n    void metoda(int a, int b) {\n        int x = a * b;\n        int y = a + b;\n        int z = x + y;\n        std::cout << \"Wynik: \" << z << std::endl;\n    }\n};\n";
            }
        }
    }

    return ss.str();
}

bool validate_code(const std::string& modified_code) {
    std::ofstream file("modified_code.cpp", std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku do zapisu zmodyfikowanego kodu.\n");
        return false;
    }
    file << "\xEF\xBB\xBF"; // Dodanie BOM (Byte Order Mark) dla UTF-8
    file << modified_code;
    file.close();

    // Kompilacja kodu
    std::string compile_command = "g++ -o modified_code.exe modified_code.cpp -I C:/Users/MOREK/source/repos/WindowsProject1/vcpkg/installed/x64-windows/include/Eigen/Dense";
    int compile_result = std::system(compile_command.c_str());
    if (compile_result != 0) {
        LogErrorToEditControl(hwndErrorEdit, L"Błąd kompilacji zmodyfikowanego kodu.\n");
        return false;
    }

    // Analiza statyczna kodu
    std::string analysis_command = "cppcheck --enable=all modified_code.cpp";
    int analysis_result = std::system(analysis_command.c_str());
    if (analysis_result != 0) {
        LogErrorToEditControl(hwndErrorEdit, L"Błędy analizy statycznej w zmodyfikowanym kodzie.\n");
        return false;
    }

    // Analiza statyczna za pomocą clang-tidy
    std::string clang_tidy_command = "clang-tidy modified_code.cpp -- -std=c++17";
    int clang_tidy_result = std::system(clang_tidy_command.c_str());
    if (clang_tidy_result != 0) {
        LogErrorToEditControl(hwndErrorEdit, L"Błędy analizy statycznej za pomocą clang-tidy w zmodyfikowanym kodzie.\n");
        return false;
    }

    return true;
}


bool test_modified_code(const std::string& modified_code) {
    std::ofstream file("modified_code.cpp", std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku do zapisu zmodyfikowanego kodu.\n");
        return false;
    }
    file << "\xEF\xBB\xBF"; // Dodanie BOM (Byte Order Mark) dla UTF-8
    file << modified_code;
    file.close();

    // Kompilacja kodu
    std::string compile_command = "g++ -o modified_code.exe modified_code.cpp";
    int compile_result = std::system(compile_command.c_str());
    if (compile_result != 0) {
        LogErrorToEditControl(hwndErrorEdit, L"Błąd kompilacji zmodyfikowanego kodu.\n");
        return false;
    }

    // Uruchomienie testów jednostkowych
    std::string run_command = "./modified_code.exe";
    int run_result = std::system(run_command.c_str());
    if (run_result != 0) {
        LogErrorToEditControl(hwndErrorEdit, L"Błąd uruchomienia zmodyfikowanego kodu.\n");
        return false;
    }

    return true;
}


double measure_performance(const std::string& modified_code) {
    std::ofstream file("modified_code.cpp");
    if (!file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku do zapisu zmodyfikowanego kodu.\n");
        return -1.0;
    }
    file << modified_code;
    file.close();

    // Kompilacja kodu
    std::string compile_command = "g++ -o modified_code.exe modified_code.cpp";
    int compile_result = std::system(compile_command.c_str());
    if (compile_result != 0) {
        LogErrorToEditControl(hwndErrorEdit, L"Błąd kompilacji zmodyfikowanego kodu.\n");
        return -1.0;
    }

    // Pomiar czasu wykonania
    auto start = std::chrono::high_resolution_clock::now();
    std::string run_command = "./modified_code.exe";
    int run_result = std::system(run_command.c_str());
    auto end = std::chrono::high_resolution_clock::now();

    if (run_result != 0) {
        LogErrorToEditControl(hwndErrorEdit, L"Błąd uruchomienia zmodyfikowanego kodu.\n");
        return -1.0;
    }

    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

void train_with_reinforcement_learning(std::shared_ptr<RecurrentNeuralNetwork> rnn, const vector<VectorXd>& inputs, const vector<VectorXd>& targets, double initial_learning_rate, int epochs, HWND hwnd) {
    if (inputs.size() != targets.size()) {
        std::wstring error_message = L"Rozmiary inputs (" + std::to_wstring(inputs.size()) + L") i targets (" + std::to_wstring(targets.size()) + L") są niezgodne.";
        LogErrorToEditControl(hwndErrorEdit, error_message);
        return;
    }

    std::thread training_thread([=]() {
        double learning_rate = initial_learning_rate;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto start_epoch = std::chrono::high_resolution_clock::now();
            std::vector<std::future<void>> futures;
            size_t num_threads = std::thread::hardware_concurrency();
            size_t chunk_size = inputs.size() / num_threads;

            for (size_t t = 0; t < num_threads; ++t) {
                futures.push_back(std::async(std::launch::async, [&, t]() {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? inputs.size() : start + chunk_size;

                    for (size_t i = start; i < end; ++i) {
                        try {
                            int index = static_cast<int>(i);
                            validate_index(index, static_cast<int>(inputs.size()));
                            validate_index(index, static_cast<int>(targets.size()));
                            if (inputs[i].size() != rnn->get_input_size()) {
                                std::wstring error_message = L"Rozmiar wejścia (" + std::to_wstring(inputs[i].size()) + L") jest niezgodny z oczekiwanym rozmiarem sieci (" + std::to_wstring(rnn->get_input_size()) + L").";
                                LogErrorToEditControl(hwndErrorEdit, error_message);
                                return;
                            }

                            if (targets[i].size() != rnn->get_output_size()) {
                                std::wstring error_message = L"Rozmiar celu (" + std::to_wstring(targets[i].size()) + L") jest niezgodny z oczekiwanym rozmiarem wyjścia sieci (" + std::to_wstring(rnn->get_output_size()) + L").";
                                LogErrorToEditControl(hwndErrorEdit, error_message);
                                return;
                            }

                            // Sprawdzenie wyrównania pamięci
                            if (reinterpret_cast<uintptr_t>(inputs[i].data()) % 16 != 0) {
                                LogErrorToEditControl(hwndErrorEdit, L"Dane wejściowe nie są wyrównane do 16 bajtów!\n");
                                return;
                            }

                            // Sprawdzenie wskaźnika
                            if (inputs[i].data() == nullptr) {
                                LogErrorToEditControl(hwndErrorEdit, L"Wskaźnik inputs[i].data() jest nullptr!\n");
                                return;
                            }

                            VectorXd output = rnn->forward(inputs[i]);
                            VectorXd reward = targets[i] - output;

                            if (reward.size() != rnn->get_output_size()) {
                                std::wstring error_message = L"Rozmiar reward (" + std::to_wstring(reward.size()) + L") jest niezgodny z oczekiwanym rozmiarem wyjścia sieci (" + std::to_wstring(rnn->get_output_size()) + L").";
                                LogErrorToEditControl(hwndErrorEdit, error_message);
                                return;
                            }

                            rnn->train({ inputs[i] }, { reward }, learning_rate);

                            double loss = reward.squaredNorm();
                            double accuracy = (reward.array().abs() < 0.5).cast<double>().mean();
                            AddDataPoint(loss, accuracy);
                        }
                        catch (const std::invalid_argument& e) {
                            std::wstring error_message = L"Wyjątek std::invalid_argument: " + std::wstring(e.what(), e.what() + strlen(e.what()));
                            LogErrorToEditControl(hwndErrorEdit, error_message);
                        }
                        catch (const std::exception& e) {
                            std::wstring error_message = L"Standardowy wyjątek: " + std::wstring(e.what(), e.what() + strlen(e.what()));
                            LogErrorToEditControl(hwndErrorEdit, error_message);
                        }
                        catch (...) {
                            LogErrorToEditControl(hwndErrorEdit, L"Nieznany wyjątek.");
                        }
                    }
                    }));
            }

            for (auto& future : futures) {
                future.get();
            }

            auto end_epoch = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> epoch_duration = end_epoch - start_epoch;
            {
                std::wstringstream ws;
                ws << L"Epoch " << epoch + 1 << L" completed in " << epoch_duration.count() << L" seconds.\n";
                LogToEditControl(hwndErrorEdit, ws.str());
            }

            {
                std::lock_guard<std::mutex> lock(mtx);
                if (IsWindow(hwnd)) {
                    PostMessage(hwnd, WM_USER + 1, static_cast<WPARAM>(epoch + 1), 0);
                }
            }

            {
                std::lock_guard<std::mutex> lock(cv_mtx);
                if (!training_in_progress) {
                    LogToEditControl(hwndErrorEdit, L"Trening przerwany.\n");
                    return;
                }
            }

            // Zmniejszanie współczynnika uczenia co 10 epok
            if ((epoch + 1) % 10 == 0) {
                learning_rate *= 0.9;
            }

            // Generowanie modyfikacji kodu po każdej epoce
            int attempts = 0; // Inicjalizacja zmiennej attempts

            VectorXd input = reprezentacja_wektorowa(kod_zrodlowy);
            VectorXd output = rnn->forward(input);
            std::string zmodyfikowany_kod = przekształć_na_kod_zrodlowy(output, attempts); // Dodano brakujący argument attempts

            // Testowanie zmodyfikowanego kodu
            if (validate_code(zmodyfikowany_kod) && test_modified_code(zmodyfikowany_kod)) {
                double performance = measure_performance(zmodyfikowany_kod);
                {
                    std::wstringstream ws;
                    ws << L"Wydajność zmodyfikowanego kodu: " << performance << L" sekund\n";
                    LogToEditControl(hwndErrorEdit, ws.str());
                }

                // Aktualizacja tekstu w kontrolce EDIT
                updateEditText(hwndEdit, zmodyfikowany_kod);

                // Aktualizacja interfejsu użytkownika po modyfikacji kodu
                updateUIAfterModification(hwndEdit, zmodyfikowany_kod);


                // Wywołanie funkcji run_logical_tests
                if (run_logical_tests(zmodyfikowany_kod)) {
                    LogToEditControl(hwndErrorEdit, L"Testy logiczne przeszły pomyślnie.\n");
                }
                else {
                    LogErrorToEditControl(hwndErrorEdit, L"Testy logiczne nie powiodły się.\n");
                }
            }
            else {
                LogErrorToEditControl(hwndErrorEdit, L"Walidacja lub testowanie zmodyfikowanego kodu nie powiodło się.\n");
            }
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            if (IsWindow(hwnd)) {
                PostMessage(hwnd, WM_USER + 2, 0, 0);
            }
        }

        // Zapisanie modelu po zakończeniu treningu
        rnn->save_model("rnn_model.bin");

        // Wyświetlenie wyników treningu
        DisplayTrainingResults(hwnd);

        // Wywołanie run_inference_on_npu po wytrenowaniu modelu
        std::vector<VectorXd> dummy_inputs = generuj_dane_treningowe(1); // Przykładowe dane wejściowe
        run_inference_on_npu("model.xml", "model.bin", dummy_inputs, hwndErrorEdit);
        });

    training_thread.detach();
}

void normalize_targets(vector<VectorXd>& targets, size_t desired_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);

    while (targets.size() < desired_size) {
        VectorXd target(10);
        for (int i = 0; i < 10; ++i) {
            target[i] = dis(gen);
        }
        targets.push_back(target);
    }
}


vector<VectorXd> przygotuj_dane_wyjsciowe(const string& pozadane_modyfikacje) {
    vector<VectorXd> targets;

    // Dzieli ciąg na pojedyncze modyfikacje
    istringstream iss(pozadane_modyfikacje);
    string modyfikacja;
    while (getline(iss, modyfikacja, ',')) {
        VectorXd target = VectorXd::Zero(10); // Inicjalizacja wektora zerami

        // Parsowanie modyfikacji za pomocą wyrażeń regularnych
        smatch match;
        try {
            if (regex_search(modyfikacja, match, regex(R"(Dodaj funkcję (\d+))"))) {
                int indeks = stoi(match[1]);
                validate_index(indeks, 10);
                target[indeks] = 1.0;
            }
            else if (regex_search(modyfikacja, match, regex(R"(Usuń funkcję (\d+))"))) {
                int indeks = stoi(match[1]);
                validate_index(indeks, 10);
                target[indeks] = -1.0;
            }
            else if (regex_search(modyfikacja, match, regex(R"(Zmień parametry funkcji (\d+))"))) {
                int indeks = stoi(match[1]);
                validate_index(indeks, 10);
                target[indeks] = 0.5; // Przykładowa wartość dla zmiany parametrów funkcji
            }
            else if (regex_search(modyfikacja, match, regex(R"(Dodaj klasę (\d+))"))) {
                int indeks = stoi(match[1]);
                validate_index(indeks, 10);
                target[indeks] = 2.0; // Przykładowa wartość dla dodania klasy
            }
            else if (regex_search(modyfikacja, match, regex(R"(Usuń klasę (\d+))"))) {
                int indeks = stoi(match[1]);
                validate_index(indeks, 10);
                target[indeks] = -2.0; // Przykładowa wartość dla usunięcia klasy
            }
            else {
                throw invalid_argument("Nieznana modyfikacja: " + modyfikacja);
            }
        }
        catch (const exception& e) {
            std::wstringstream ws;
            ws << L"Błąd parsowania modyfikacji: " << e.what();
            LogErrorToEditControl(hwndErrorEdit, ws.str());
            continue; // Pomijamy niepoprawne modyfikacje
        }

        targets.push_back(target);
    }

    // Uzupełnienie do 1000 wektorów wyjściowych, jeśli jest ich mniej
    normalize_targets(targets, 1000);

    return targets;
}

vector<VectorXd> wczytaj_dane_wejsciowe_z_pliku(const string& nazwa_pliku) {
    vector<VectorXd> dane_wejsciowe;
    ifstream plik(nazwa_pliku);
    if (!plik.is_open()) {
        std::wstringstream ws;
        ws << L"Nie można otworzyć pliku: " << std::wstring(nazwa_pliku.begin(), nazwa_pliku.end());
        LogErrorToEditControl(hwndErrorEdit, ws.str());
        return dane_wejsciowe; // Zwracamy pusty wektor w przypadku błędu
    }

    string linia;
    while (getline(plik, linia)) {
        stringstream ss(linia);
        double liczba;
        VectorXd wektor(224 * 224 * 3); // Zakładamy, że wektor ma rozmiar 224 * 224 * 3
        int indeks = 0;
        while (ss >> liczba && indeks < wektor.size()) {
            validate_index(indeks, static_cast<int>(wektor.size()));
            wektor[indeks++] = liczba;

        }
        if (indeks == wektor.size()) {
            // Sprawdzenie wyrównania pamięci
            check_alignment(wektor.data(), 16);
            dane_wejsciowe.push_back(wektor);
            manual_data_count++;
            total_data_count++;
        }
        else {
            std::wstringstream ws;
            ws << L"Nieprawidłowy format linii w pliku: " << std::wstring(linia.begin(), linia.end());
            LogErrorToEditControl(hwndErrorEdit, ws.str());
        }
    }

    plik.close();
    return dane_wejsciowe;
}



vector<VectorXd> przygotuj_dane_wejsciowe(const string& kod_zrodlowy, int input_size) {
    vector<VectorXd> inputs;

    // Inicjalizacja Clang
    CXIndex index = clang_createIndex(0, 0);

    // Ustawienia opcji parsowania
    unsigned int options = CXTranslationUnit_None;

    // Tworzenie tymczasowego pliku z kodem źródłowym
    std::ofstream temp_file("temp_code.cpp");
    if (!temp_file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku temp_code.cpp do zapisu.\n");
        return inputs;
    }
    temp_file << kod_zrodlowy;
    temp_file.close();

    // Parsowanie jednostki translacji
    CXTranslationUnit unit = clang_parseTranslationUnit(
        index, "temp_code.cpp", nullptr, 0, nullptr, 0, options);

    if (unit == nullptr) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można sparsować kodu źródłowego.\n");
        clang_disposeIndex(index);
        return inputs;
    }

    // Przechodzenie po drzewie AST i ekstrakcja cech
    CXCursor cursor = clang_getTranslationUnitCursor(unit);
    VectorXd cechy = VectorXd::Zero(input_size);
    int feature_index = 0;

    auto data = std::make_pair(&cechy, &feature_index);

    clang_visitChildren(
        cursor,
        [](CXCursor c, CXCursor parent, CXClientData client_data) {
            auto data = static_cast<std::pair<VectorXd*, int*>*>(client_data);
            VectorXd* cechy = data->first;
            int* feature_index = data->second;

            if (clang_Location_isFromMainFile(clang_getCursorLocation(c)) == 0) {
                return CXChildVisit_Continue;
            }

            // Ekstrakcja cech na podstawie typu kursora
            CXCursorKind kind = clang_getCursorKind(c);
            if (kind == CXCursor_FunctionDecl || kind == CXCursor_VarDecl || kind == CXCursor_ParmDecl) {
                if (*feature_index < cechy->size()) {
                    (*cechy)(*feature_index) = 1.0;
                    (*feature_index)++;
                }
            }

            return CXChildVisit_Recurse;
        },
        &data);

    // Czyszczenie zasobów Clang
    clang_disposeTranslationUnit(unit);
    clang_disposeIndex(index);

    // Usunięcie tymczasowego pliku
    std::remove("temp_code.cpp");

    // Generowanie danych wejściowych na podstawie wektora cech
    for (int i = 0; i < 1000; ++i) { // Zwiększenie liczby danych testowych
        VectorXd input = cechy + VectorXd::Random(input_size) * 0.01; // Dodanie losowego szumu
        inputs.push_back(input);
    }

    return inputs;
}


// Funkcja generująca modyfikacje na podstawie wyników modelu
std::string generuj_modyfikacje(const VectorXd& output) {
    std::wstringstream ws;
    for (int i = 0; i < output.size(); ++i) {
        if (output[i] > 0.7) {
            ws << L"Dodaj funkcję o indeksie " << i << L" z parametrami (int a, int b);\n";
        }
        else if (output[i] > 0.5) {
            ws << L"Zmień parametry funkcji o indeksie " << i << L" na (double x, double y);\n";
        }
        else if (output[i] > 0.3) {
            ws << L"Dodaj klasę o indeksie " << i << L" z metodą void metoda();\n";
        }
        else if (output[i] > 0.1) {
            ws << L"Dodaj zmienną globalną int zmienna_" << i << L";\n";
        }
        else if (output[i] > -0.1) {
            ws << L"Usuń klasę o indeksie " << i << L";\n";
        }
        else if (output[i] > -0.3) {
            ws << L"Usuń zmienną globalną int zmienna_" << i << L";\n";
        }
        else if (output[i] > -0.5) {
            ws << L"Zmień typ zmiennej globalnej int zmienna_" << i << L" na double;\n";
        }
        else {
            ws << L"Usuń funkcję o indeksie " << i << L";\n";
        }
    }
    LogToEditControl(hwndErrorEdit, ws.str());

    return WideStringToString(ws.str());
}


// Funkcja aplikująca modyfikacje do kodu źródłowego
// Funkcja do obliczania złożoności kodu za pomocą cppcheck
double calculate_complexity(const std::string& code) {
    std::ofstream file("temp_code.cpp");
    if (!file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku do zapisu kodu.\n");
        return 0.0;
    }
    file << code;
    file.close();

    std::string command = "cppcheck --enable=all --xml --xml-version=2 temp_code.cpp 2> cppcheck_results.xml";
    std::system(command.c_str());

    std::ifstream result_file("cppcheck_results.xml");
    if (!result_file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku z wynikami cppcheck.\n");
        return 0.0;
    }

    std::stringstream buffer;
    buffer << result_file.rdbuf();
    std::string results = buffer.str();
    result_file.close();

    // Analiza wyników cppcheck w celu obliczenia złożoności
    size_t complexity_pos = results.find("complexity=");
    if (complexity_pos != std::string::npos) {
        size_t start = results.find("\"", complexity_pos) + 1;
        size_t end = results.find("\"", start);
        std::string complexity_str = results.substr(start, end - start);
        return std::stod(complexity_str);
    }

    return 0.0;
}

// Funkcja aplikująca modyfikacje do kodu źródłowego
void aplikuj_modyfikacje(const std::string& kod_zrodlowy, std::string& zmodyfikowany_kod) {
    std::lock_guard<std::mutex> lock(kod_mtx);

    // Tworzenie tymczasowego pliku z kodem źródłowym
    std::ofstream temp_file("temp_code.cpp");
    if (!temp_file.is_open()) {
        std::cerr << "Nie można otworzyć pliku temp_code.cpp do zapisu.\n";
        return;
    }
    temp_file << kod_zrodlowy;
    temp_file.close();

    // Inicjalizacja Clang
    CXIndex index = clang_createIndex(0, 0);
    unsigned int options = CXTranslationUnit_None;
    CXTranslationUnit unit = clang_parseTranslationUnit(index, "temp_code.cpp", nullptr, 0, nullptr, 0, options);
    if (unit == nullptr) {
        std::cerr << "Nie można sparsować kodu źródłowego.\n";
        clang_disposeIndex(index);
        return;
    }

    // Przechodzenie po drzewie AST i stosowanie modyfikacji
    CXCursor cursor = clang_getTranslationUnitCursor(unit);
    std::string nowy_kod = kod_zrodlowy; // Użycie lokalnej zmiennej do przechowywania zmodyfikowanego kodu
    clang_visitChildren(
        cursor,
        [](CXCursor c, CXCursor parent, CXClientData client_data) {
            std::string* nowy_kod = static_cast<std::string*>(client_data);
            CXCursorKind kind = clang_getCursorKind(c);

            if (kind == CXCursor_FunctionDecl) {
                CXString name = clang_getCursorSpelling(c);
                std::string function_name = clang_getCString(name);
                clang_disposeString(name);

                // Przykład modyfikacji: zmiana nazwy funkcji
                if (function_name == "stara_funkcja") {
                    std::regex re("void stara_funkcja\\(");
                    *nowy_kod = std::regex_replace(*nowy_kod, re, "void nowa_funkcja(");
                }
            }
            return CXChildVisit_Recurse;
        },
        static_cast<void*>(&nowy_kod));

    // Czyszczenie zasobów Clang
    clang_disposeTranslationUnit(unit);
    clang_disposeIndex(index);

    // Aktualizacja zmodyfikowanego kodu
    zmodyfikowany_kod = nowy_kod;
}


// Funkcja do oceny czytelności kodu za pomocą clang-tidy
double calculate_readability(const std::string& code) {
    std::ofstream file("temp_code.cpp");
    if (!file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku do zapisu kodu.\n");
        return 0.0;
    }
    file << code;
    file.close();

    std::string command = "clang-tidy temp_code.cpp -- -std=c++17 > clang_tidy_results.txt";
    std::system(command.c_str());

    std::ifstream result_file("clang_tidy_results.txt");
    if (!result_file.is_open()) {
        LogErrorToEditControl(hwndErrorEdit, L"Nie można otworzyć pliku z wynikami clang-tidy.\n");
        return 0.0;
    }

    std::stringstream buffer;
    buffer << result_file.rdbuf();
    std::string results = buffer.str();
    result_file.close();

    // Analiza wyników clang-tidy w celu oceny czytelności
    size_t readability_pos = results.find("readability-");
    if (readability_pos != std::string::npos) {
        return 1.0; // Zakładamy, że znalezienie problemów z czytelnością oznacza niską czytelność
    }

    return 0.0; // Brak problemów z czytelnością
}

// Struktura przechowująca wyniki oceny
struct EvaluationResult {
    double validation_score;
    double test_score;
    double performance_score;
    double logical_test_score;
    double complexity_score;
    double readability_score;
    double total_score;
    double performance_improvement;
    std::vector<std::string> static_analysis_issues;
    double logical_test_coverage;

    // Konstruktor inicjalizujący wszystkie zmienne składowe
    EvaluationResult()
        : validation_score(0.0),
        test_score(0.0),
        performance_score(0.0),
        logical_test_score(0.0),
        complexity_score(0.0),
        readability_score(0.0),
        total_score(0.0),
        performance_improvement(0.0),
        logical_test_coverage(0.0) {
    }
};

// Struktura konfiguracji
struct Config {
    int max_attempts;
    double min_performance_increase;
    double max_execution_time;
    double validation_weight;
    double test_weight;
    double performance_weight;
    double logical_test_weight;
    std::string retry_strategy;
    double complexity_weight;
    double readability_weight;
    std::string config_path;

    // Konstruktor inicjalizujący wszystkie zmienne składowe
    Config()
        : max_attempts(10),
        min_performance_increase(0.1),
        max_execution_time(5.0),
        validation_weight(0.4),
        test_weight(0.4),
        performance_weight(0.2),
        logical_test_weight(0.1),
        retry_strategy("minor_changes"),
        complexity_weight(0.1),
        readability_weight(0.1),
        config_path("C:/Users/MOREK/source/repos/WindowsProject1/WindowsProject1/config.json") {
    }
};

// Funkcja do wczytywania konfiguracji z pliku JSON
Config loadConfigFromFile(const std::string& filename) {
    Config config;
    std::ifstream file(filename);
    if (file.is_open()) {
        nlohmann::json jsonConfig;
        file >> jsonConfig;
        config.max_attempts = jsonConfig.value("max_attempts", 10);
        config.min_performance_increase = jsonConfig.value("min_performance_increase", 0.1);
        config.max_execution_time = jsonConfig.value("max_execution_time", 5.0);
        config.validation_weight = jsonConfig.value("validation_weight", 0.4);
        config.test_weight = jsonConfig.value("test_weight", 0.4);
        config.performance_weight = jsonConfig.value("performance_weight", 0.2);
        config.logical_test_weight = jsonConfig.value("logical_test_weight", 0.1); // Dodano wczytywanie wagi dla testów logicznych
        config.retry_strategy = jsonConfig.value("retry_strategy", "minor_changes"); // Dodano wczytywanie strategii ponawiania
        config.complexity_weight = jsonConfig.value("complexity_weight", 0.1); // Dodano wczytywanie wagi dla złożoności
        config.readability_weight = jsonConfig.value("readability_weight", 0.1); // Dodano wczytywanie wagi dla czytelności
        config.config_path = jsonConfig.value("config_path", "C:/Users/MOREK/source/repos/WindowsProject1/WindowsProject1/config.json"); // Domyślna wartość ścieżki
    }
    else {
        // Ustawienia domyślne w przypadku braku pliku konfiguracyjnego
        config.max_attempts = 10;
        config.min_performance_increase = 0.1;
        config.max_execution_time = 5.0;
        config.validation_weight = 0.4;
        config.test_weight = 0.4;
        config.performance_weight = 0.2;
        config.logical_test_weight = 0.1; // Domyślna wartość wagi dla testów logicznych
        config.retry_strategy = "minor_changes"; // Domyślna wartość strategii ponawiania
        config.complexity_weight = 0.1; // Domyślna wartość wagi dla złożoności
        config.readability_weight = 0.1; // Domyślna wartość wagi dla czytelności
        config.config_path = "C:/Users/MOREK/source/repos/WindowsProject1/WindowsProject1/config.json"; // Domyślna wartość ścieżki
    }
    return config;
}




// Funkcja do obliczania ocen cząstkowych
EvaluationResult evaluate_modification(const std::string& modified_code, const Config& config, double original_performance) {
    EvaluationResult result;
    result.validation_score = validate_code(modified_code) ? 1.0 : 0.0;
    result.test_score = test_modified_code(modified_code) ? 1.0 : 0.0;
    result.performance_score = measure_performance(modified_code);
    result.logical_test_score = run_logical_tests(modified_code) ? 1.0 : 0.0;
    result.complexity_score = calculate_complexity(modified_code);
    result.readability_score = calculate_readability(modified_code);
    result.performance_improvement = result.performance_score - original_performance;
    result.logical_test_coverage = 1.0; // Placeholder for logical test coverage
    result.static_analysis_issues = {}; // Placeholder for static analysis issues

    result.total_score = config.validation_weight * result.validation_score +
        config.test_weight * result.test_score +
        config.performance_weight * result.performance_score +
        config.logical_test_weight * result.logical_test_score +
        config.complexity_weight * result.complexity_score +
        config.readability_weight * result.readability_score;

    return result;
}


// Funkcja do automatycznego generowania i aplikowania modyfikacji na podstawie wyników modelu
void automatyczne_aplikowanie_modyfikacji(const std::string& kod_zrodlowy) {
    Config config = loadConfigFromFile("config.json");

    int attempts = 0;
    double original_performance = measure_performance(kod_zrodlowy);
    double best_performance = original_performance;
    std::string best_code = kod_zrodlowy;
    double best_score = 0.0;
    int no_improvement_attempts = 0;

    while (attempts < config.max_attempts) {
        attempts++;
        std::wstringstream log_message;
        log_message << L"Próba numer: " << attempts << L"\n";
        LogErrorToEditControl(hwndErrorEdit, log_message.str());

        VectorXd input = reprezentacja_wektorowa(kod_zrodlowy);
        auto rnn = std::make_shared<RecurrentNeuralNetwork>(input.size(), 128, 10);
        rnn->load_model("rnn_model.bin");
        VectorXd output = rnn->forward(input);

        std::string zmodyfikowany_kod = przekształć_na_kod_zrodlowy(output, attempts);
        aplikuj_modyfikacje(kod_zrodlowy, zmodyfikowany_kod);

        if (!validate_code(zmodyfikowany_kod)) {
            LogErrorToEditControl(hwndErrorEdit, L"Walidacja zmodyfikowanego kodu nie powiodła się.\n");
            continue;
        }

        if (!test_modified_code(zmodyfikowany_kod)) {
            LogErrorToEditControl(hwndErrorEdit, L"Testowanie zmodyfikowanego kodu nie powiodło się.\n");
            continue;
        }

        double performance = measure_performance(zmodyfikowany_kod);
        if (performance < 0 || performance > config.max_execution_time) {
            LogErrorToEditControl(hwndErrorEdit, L"Mierzenie wydajności zmodyfikowanego kodu nie powiodło się lub przekroczyło maksymalny czas wykonania.\n");
            continue;
        }

        std::wstringstream ws;
        ws << L"Zmodyfikowany kod został pomyślnie wygenerowany, przetestowany i zmierzony.\n";
        ws << L"Wydajność zmodyfikowanego kodu: " << performance << L" sekund\n";
        LogErrorToEditControl(hwndErrorEdit, ws.str());

        if (run_logical_tests(zmodyfikowany_kod)) {
            LogErrorToEditControl(hwndErrorEdit, L"Testy logiczne przeszły pomyślnie.\n");
        }
        else {
            LogErrorToEditControl(hwndErrorEdit, L"Testy logiczne nie powiodły się.\n");
            continue;
        }

        EvaluationResult result = evaluate_modification(zmodyfikowany_kod, config, original_performance);

        // Złożona logika warunkowa
        if (result.total_score > best_score) {
            best_score = result.total_score;
            best_performance = performance;
            best_code = zmodyfikowany_kod;
            no_improvement_attempts = 0;
        }
        else {
            no_improvement_attempts++;
        }

        // Priorytetyzacja kryteriów
        if (result.performance_improvement > config.min_performance_increase) {
            best_score += config.performance_weight * result.performance_score;
        }

        // Strategie decyzyjne
        if (config.retry_strategy == "aggressive") {
            if (result.performance_improvement > 0.2) {
                best_score += 0.1;
            }
        }
        else if (config.retry_strategy == "conservative") {
            if (result.performance_improvement < 0.1) {
                best_score -= 0.1;
            }
        }

        if (no_improvement_attempts >= config.max_attempts / 2) {
            LogErrorToEditControl(hwndErrorEdit, L"Wczesne zatrzymanie: brak znaczącej poprawy po określonej liczbie prób.\n");
            break;
        }
    }

    std::wstringstream final_report;
    final_report << L"Podsumowanie:\n";
    final_report << L"Najlepsza wydajność: " << best_performance << L" sekund\n";
    final_report << L"Najlepszy kod:\n" << StringToWideString(best_code) << L"\n";
    final_report << L"Ocena: " << best_score << L"\n";
    LogErrorToEditControl(hwndErrorEdit, final_report.str());

    updateUIAfterModification(hwndEdit, best_code);
}

void apply_rnn_modifications(const std::string& kod_zrodlowy) {
    int attempts = 0; // Inicjalizacja zmiennej attempts

    // Generowanie wektora cech na podstawie kodu źródłowego
    VectorXd input = reprezentacja_wektorowa(kod_zrodlowy);

    // Wywołanie modelu RNN
    auto rnn = std::make_shared<RecurrentNeuralNetwork>(input.size(), 128, 10);
    rnn->load_model("rnn_model.bin");
    VectorXd output = rnn->forward(input);

    // Generowanie modyfikacji na podstawie wyjścia modelu RNN
    std::string zmodyfikowany_kod = przekształć_na_kod_zrodlowy(output, attempts); // Dodano brakujący argument attempts

    // Aplikowanie modyfikacji do kodu źródłowego
    aplikuj_modyfikacje(kod_zrodlowy, zmodyfikowany_kod);

    // Walidacja zmodyfikowanego kodu
    if (!validate_code(zmodyfikowany_kod)) {
        LogErrorToEditControl(hwndErrorEdit, L"Walidacja zmodyfikowanego kodu nie powiodła się.\n");
        return;
    }

    // Testowanie zmodyfikowanego kodu
    if (!test_modified_code(zmodyfikowany_kod)) {
        LogErrorToEditControl(hwndErrorEdit, L"Testowanie zmodyfikowanego kodu nie powiodło się.\n");
        return;
    }

    // Mierzenie wydajności zmodyfikowanego kodu
    double performance = measure_performance(zmodyfikowany_kod);
    if (performance < 0) {
        LogErrorToEditControl(hwndErrorEdit, L"Mierzenie wydajności zmodyfikowanego kodu nie powiodło się.\n");
        return;
    }

    std::wstringstream ws;
    ws << L"Zmodyfikowany kod został pomyślnie wygenerowany, przetestowany i zmierzony.\n";
    ws << L"Wydajność zmodyfikowanego kodu: " << performance << L" sekund\n";
    LogToEditControl(hwndErrorEdit, ws.str());

    // Wywołanie funkcji run_logical_tests
    if (run_logical_tests(zmodyfikowany_kod)) {
        LogToEditControl(hwndErrorEdit, L"Testy logiczne przeszły pomyślnie.\n");
    }
    else {
        LogErrorToEditControl(hwndErrorEdit, L"Testy logiczne nie powiodły się.\n");
    }

    // Aktualizacja interfejsu użytkownika po modyfikacji kodu
    updateUIAfterModification(hwndEdit, zmodyfikowany_kod);
}

std::string generuj_przykladowy_kod() {
    std::stringstream ss;
    ss << "#include <iostream>\n";
    ss << "void funkcja_" << rand() % 100 << "() {\n";
    ss << "    std::wstringstream ws;\n";
    ss << "    ws << L\"Hello, World!\" << std::endl;\n";
    ss << "    LogToEditControl(hwndErrorEdit, ws.str());\n";
    ss << "}\n";
    return ss.str();
}
void save_code_to_file(const std::string& filename, const std::string& code) {
    try {
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            throw std::ios_base::failure("Nie można otworzyć pliku do zapisu kodu.");
        }
        file << code;
        file.close();
    }
    catch (const std::exception& e) {
        std::wstring error_message = L"Wyjątek podczas zapisu pliku: " + std::wstring(e.what(), e.what() + strlen(e.what()));
        LogErrorToEditControl(hwndErrorEdit, error_message);
    }
}

void integrate_workflow(HWND hwnd) {
    try {
        // Generowanie danych treningowych
        vector<VectorXd> training_data = generuj_dane_treningowe(1000);
        vector<VectorXd> targets = przygotuj_dane_wyjsciowe("przykładowe pożądane modyfikacje");

        // Inicjalizacja modelu RNN
        int input_size = 224 * 224 * 3;
        int hidden_size = 128;
        int output_size = 10;
        auto rnn = std::make_shared<RecurrentNeuralNetwork>(input_size, hidden_size, output_size);

        // Parametry pętli głównej
        double learning_rate = 0.01;
        int max_iterations = 10; // Maksymalna liczba iteracji
        int epochs = 100;

        Config config = loadConfigFromFile("config.json");
        double original_performance = measure_performance(kod_zrodlowy);
        double best_performance = original_performance;
        std::string best_code = kod_zrodlowy;
        double best_score = 0.0;

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            // Trenowanie modelu RNN
            train_with_reinforcement_learning(rnn, training_data, targets, learning_rate, epochs, hwnd);

            // Pobieranie kodu źródłowego z kontrolki edycji
            int length = GetWindowTextLength(hwndEdit);
            std::wstring w_kod_zrodlowy(length, L'\0');
            GetWindowTextW(hwndEdit, &w_kod_zrodlowy[0], length + 1);
            std::string kod_zrodlowy = WideStringToString(w_kod_zrodlowy);

            // Generowanie modyfikacji kodu
            VectorXd input = reprezentacja_wektorowa(kod_zrodlowy);
            VectorXd output = rnn->forward(input);
            int attempts = 1;
            std::string zmodyfikowany_kod = przekształć_na_kod_zrodlowy(output, attempts);

            // Walidacja zmodyfikowanego kodu
            if (!validate_code(zmodyfikowany_kod)) {
                LogErrorToEditControl(hwndErrorEdit, L"Walidacja zmodyfikowanego kodu nie powiodła się.\n");
                continue;
            }

            // Testowanie zmodyfikowanego kodu
            if (!test_modified_code(zmodyfikowany_kod)) {
                LogErrorToEditControl(hwndErrorEdit, L"Testowanie zmodyfikowanego kodu nie powiodło się.\n");
                continue;
            }

            // Mierzenie wydajności zmodyfikowanego kodu
            double performance = measure_performance(zmodyfikowany_kod);
            if (performance < 0 || performance > config.max_execution_time) {
                LogErrorToEditControl(hwndErrorEdit, L"Mierzenie wydajności zmodyfikowanego kodu nie powiodło się lub przekroczyło maksymalny czas wykonania.\n");
                continue;
            }

            // Wywołanie funkcji run_logical_tests
            if (!run_logical_tests(zmodyfikowany_kod)) {
                LogErrorToEditControl(hwndErrorEdit, L"Testy logiczne nie powiodły się.\n");
                continue;
            }

            // Ocena zmodyfikowanego kodu
            EvaluationResult result = evaluate_modification(zmodyfikowany_kod, config, original_performance);

            // Decyzja o zastosowaniu zmodyfikowanego kodu
            if (result.total_score > best_score) {
                best_score = result.total_score;
                best_performance = performance;
                best_code = zmodyfikowany_kod;
            }

            // Aktualizacja tekstu w kontrolce EDIT
            updateEditText(hwndEdit, best_code);

            // Uruchamianie inferencji na modelu NPU
            std::vector<VectorXd> dummy_inputs = generuj_dane_treningowe(1);
            run_inference_on_npu("model.xml", "model.bin", dummy_inputs, hwndErrorEdit);

            // Optymalizacja modelu OpenVINO
            optimize_model_with_openvino("model.xml", "model.bin");

            // Wyświetlenie wyników treningu
            DisplayTrainingResults(hwnd);

            // Sprawdzenie warunku zakończenia (np. osiągnięcie zadowalającego poziomu optymalizacji)
            if (result.total_score >= config.min_performance_increase) {
                LogToEditControl(hwndErrorEdit, L"Osiągnięto zadowalający poziom optymalizacji.\n");
                break;
            }
        }

        // Finalna aktualizacja interfejsu użytkownika po zakończeniu pętli
        updateUIAfterModification(hwndEdit, best_code);

        // Automatyczne zapisywanie zmodyfikowanego kodu do pliku
        save_code_to_file("modified_code.cpp", best_code);
    }
    catch (const std::exception& e) {
        std::wstring error_message = L"Wyjątek podczas integracji workflow: " + std::wstring(e.what(), e.what() + strlen(e.what()));
        LogErrorToEditControl(hwndErrorEdit, error_message);
    }
}


int WINAPI wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow) {
    // Inicjalizacja GDI+
    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    // Rejestracja klasy okna
    const wchar_t CLASS_NAME[] = L"Sample Window Class";
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    RegisterClassW(&wc);

    // Tworzenie okna
    hMainWindow = CreateWindowExW(
        0,
        CLASS_NAME,
        L"Trenowanie modelu",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    if (hMainWindow == NULL) {
        return 0;
    }

    ShowWindow(hMainWindow, nCmdShow);

    // Inicjalizacja danych treningowych
    initializeTrainingData();

    // Inicjalizacja monitorowania
    initializeMonitoring();

    // Generowanie kodu źródłowego
    std::string kod_zrodlowy = generuj_przykladowy_kod();
    std::string pozadane_modyfikacje = "przykładowe pożądane modyfikacje";

    int input_size = 224 * 224 * 3; // 150528
    int hidden_size = 128;
    int output_size = 10;
    vector<VectorXd> inputs = przygotuj_dane_wejsciowe(kod_zrodlowy, input_size);
    vector<VectorXd> targets = przygotuj_dane_wyjsciowe(pozadane_modyfikacje);

    // Trenowanie modelu RNN
    double learning_rate = 0.01;
    int epochs = 100;
    auto rnn = std::make_shared<RecurrentNeuralNetwork>(input_size, hidden_size, output_size);

    // Uruchomienie trenowania w osobnym wątku
    std::thread training_thread([&]() {
        train_with_reinforcement_learning(rnn, inputs, targets, learning_rate, epochs, hMainWindow);

        // Generowanie modyfikacji kodu
        VectorXd input = reprezentacja_wektorowa(kod_zrodlowy);
        VectorXd output = rnn->forward(input);
        int attempts = 1; // Liczba prób
        std::string zmodyfikowany_kod = przekształć_na_kod_zrodlowy(output, attempts);

        // Aktualizacja tekstu w kontrolce EDIT
        updateEditText(hwndEdit, zmodyfikowany_kod);
        });

    // Wątek do wczytywania modelu CNN
    std::thread load_model_thread([&]() {
        ov::Core core;
        std::shared_ptr<ov::Model> cnn_model;
        load_cnn_model("model.xml", "model.bin", core, cnn_model);
        });

    // Wątek do uruchamiania inferencji na modelu CNN
    std::thread inference_thread([&]() {
        std::shared_ptr<ov::Model> cnn_model;
        std::vector<float> input_data; // Przykładowe dane wejściowe
        std::vector<float> output_data;
        run_cnn_inference(cnn_model, input_data, output_data);
        });

    // Wątek do optymalizacji modelu OpenVINO
    std::thread optimize_model_thread([&]() {
        optimize_model_with_openvino("model.xml", "model.bin");
        });

    // Detach wątki
    training_thread.detach();
    load_model_thread.detach();
    inference_thread.detach();
    optimize_model_thread.detach();

    // Przykład modyfikacji kodu
    {
        std::lock_guard<std::mutex> lock(kod_mtx);
        kod_zrodlowy = "Nowy kod źródłowy";
        kod_zmieniony = true; // Ustawienie flagi zmiany kodu
    }

    // Pętla komunikatów
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    GdiplusShutdown(gdiplusToken);
    return 0;
}
