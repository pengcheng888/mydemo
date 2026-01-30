add_rules("mode.debug", "mode.release")
add_requires("cli11")
add_requires("pybind11")

target("test_gemm")
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

    add_files(os.projectdir().."/test_gemm.cpp")
    set_installdir(INFINI_ROOT)
target_end()


target("_infinidemo")
    set_kind("shared")
    set_default(true)
    set_languages("cxx17")
    set_warnings("all", "error")
    set_targetdir("$(buildir)")

    -- Add pybind11 package (automatically configures Python paths)
    add_packages("pybind11")
    
    -- Add source files
    -- Add model implementation files
    add_files("cmodels/resnet/modeling_resnet.cpp")
    add_files("cmodels/mnist/modeling_mnist.cpp")
    add_files("cmodels/bindings.cpp")
    
    -- Add include directories
    add_includedirs("cmodels/resnet", { public = false })
    add_includedirs("cmodels/mnist", { public = false })
    add_includedirs("cmodels", { public = false })
    
    -- add_rules("python.module", {soabi = true})
    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    add_includedirs("include", { public = false })
    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt")
    
    -- Set output filename without lib prefix (Python modules don't use lib prefix)
    if is_plat("windows") then
        set_filename("_infinidemo.pyd")
    else
        set_filename("_infinidemo.so")
    end

target_end()