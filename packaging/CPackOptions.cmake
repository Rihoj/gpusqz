# CPACK_PROJECT_CONFIG_FILE: settings that differ per package generator.

if(CPACK_GENERATOR STREQUAL "productbuild")
  # The macOS .pkg installs into its own prefix and links the programs into
  # /usr/local/bin (packaging/macos-postinstall.sh). The TGZ keeps the plain
  # bin/ lib/ share/ layout.
  set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/gpusqz")
else()
  # Only productbuild needs the "gpusqz" install component (see
  # CMakeLists.txt); every other package is one piece with no component
  # selection (WiX would otherwise show a feature tree).
  set(CPACK_MONOLITHIC_INSTALL ON)
endif()
