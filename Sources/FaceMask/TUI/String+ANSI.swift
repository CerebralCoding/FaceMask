import Foundation

let ESC = Character("\u{1B}")
let CSI = "\(ESC)["

// MARK: - SGR

public extension String {
    var bold: String {
        "\(CSI)1m\(self)\(CSI)0m"
    }

    var faint: String {
        "\(CSI)2m\(self)\(CSI)0m"
    }

    var italic: String {
        "\(CSI)3m\(self)\(CSI)0m"
    }

    var underline: String {
        "\(CSI)4m\(self)\(CSI)0m"
    }
}

// MARK: - Colors

public extension String {
    var black: String {
        "\(CSI)30m\(self)\(CSI)0m"
    }

    var red: String {
        "\(CSI)31m\(self)\(CSI)0m"
    }

    var green: String {
        "\(CSI)32m\(self)\(CSI)0m"
    }

    var yellow: String {
        "\(CSI)33m\(self)\(CSI)0m"
    }

    var blue: String {
        "\(CSI)34m\(self)\(CSI)0m"
    }

    var magenta: String {
        "\(CSI)35m\(self)\(CSI)0m"
    }

    var cyan: String {
        "\(CSI)36m\(self)\(CSI)0m"
    }

    var white: String {
        "\(CSI)37m\(self)\(CSI)0m"
    }

    var gray: String {
        "\(CSI)90m\(self)\(CSI)0m"
    }

    var brightRed: String {
        "\(CSI)91m\(self)\(CSI)0m"
    }

    var brightGreen: String {
        "\(CSI)92m\(self)\(CSI)0m"
    }

    var brightYellow: String {
        "\(CSI)93m\(self)\(CSI)0m"
    }

    var brightBlue: String {
        "\(CSI)94m\(self)\(CSI)0m"
    }

    var brightMagenta: String {
        "\(CSI)95m\(self)\(CSI)0m"
    }

    var brightCyan: String {
        "\(CSI)96m\(self)\(CSI)0m"
    }

    var brightWhite: String {
        "\(CSI)97m\(self)\(CSI)0m"
    }
}

public extension String {
    func removingSGR() -> String {
        replacingOccurrences(of: "\(ESC)\\[[0-9;]*m", with: "", options: .regularExpression)
    }
}
