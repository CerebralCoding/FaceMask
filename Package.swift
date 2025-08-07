// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "FaceMask",
    platforms: [.macOS(.v15)],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "FaceMask",
            dependencies: [],
            cxxSettings: [.define("ACCELERATE_NEW_LAPACK", to: "1"), .define("ACCELERATE_LAPACK_ILP64", to: "1")],
            linkerSettings: [.linkedFramework("Accelerate")],
        ),
    ],
)
