# CPACK_PROJECT_CONFIG_FILE: settings that differ per package generator.

# The macOS .pkg installs into its own prefix and links the programs into
# /usr/local/bin (packaging/macos-postinstall.sh). The TGZ keeps the plain
# bin/ lib/ share/ layout.
if(CPACK_GENERATOR STREQUAL "productbuild")
  set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/gpusqz")
endif()
