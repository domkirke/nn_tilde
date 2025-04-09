#pragma once
#include "m_pd.h"
#include "../../../shared/model_download.h"


namespace fs = std::filesystem; 

template <typename pd_struct>
class PdModelDownloader: public ModelDownloader {
    const pd_struct* d_parent; 

public:
    PdModelDownloader(const pd_struct* parent): d_parent(parent) {}
    PdModelDownloader(const pd_struct* parent, const std::string path): d_parent(parent) { d_path = path; }

    void print_to_parent(const std::string &message, const std::string &canal) override {
        if (d_parent != nullptr) {
            if (canal == "cout") {
                post(message.c_str()); 
            } else if (canal == "cwarn") {
                post(message.c_str()); 
            } else if (canal == "cerr") {
                pd_error(d_parent, "nn~: %s", message.c_str()); 
            }
        }
    };

    fs::path cert_path_from_path(fs::path path) {
        #if defined(_WIN32) || defined(_WIN64)
            std::string perm_path = path / "Contents" / "MacOS" / "cert.pem";
        #elif defined(__APPLE__) || defined(__MACH__)
            std::string perm_path = path / "Contents" / "MacOS" / "cert.pem";
        #elif defined(__linux__)
            std::string perm_path = path / "Contents" / "MacOS" / "cert.pem";
        #else
            std::string perm_path = "";
        #endif
        return perm_path;
    }

    void fill_dict(void* dict_to_fill) override {
        return; 
    }
};
