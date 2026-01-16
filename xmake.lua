add_rules("mode.debug", "mode.release")
add_requires("cli11")

target("example")
    set_kind("binary")
    -- add_deps("infiniop", "infinirt", "infiniccl")
    set_default(false)

    -- Add CLI11 for command line parsing
    add_packages("cli11")

    set_languages("cxx17")
    set_warnings("all", "error")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
    add_includedirs(INFINI_ROOT.."/include")
    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infinicore_cpp_api")


    -- Add InfiniCore source files for Tensor class support
    -- add_files(os.projectdir().."/src/infinicore/*.cc")
    -- add_files(os.projectdir().."/src/infinicore/context/*.cc")
    -- add_files(os.projectdir().."/src/infinicore/context/*/*.cc")
    -- add_files(os.projectdir().."/src/infinicore/tensor/*.cc")
    -- add_files(os.projectdir().."/src/infinicore/ops/*/*.cc")
    -- add_files(os.projectdir().."/src/utils/*.cc")


    add_files(os.projectdir().."/example.cpp")

    set_installdir(INFINI_ROOT)
target_end()