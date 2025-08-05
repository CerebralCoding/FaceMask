import Darwin
import Foundation

public enum Termios {
    // MARK: Public

    public static func enterRawMode() {
        var currentTermios = termios()
        tcgetattr(STDIN_FILENO, &currentTermios)
        stack.append(currentTermios)
        var rawTermios = currentTermios
        rawTermios.c_lflag &= ~UInt(ECHO | ICANON)
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &rawTermios)
    }

    public static func pop() {
        if var last = stack.popLast() {
            tcsetattr(STDIN_FILENO, TCSAFLUSH, &last)
        }
    }

    // MARK: Private

    private nonisolated(unsafe) static var stack: [termios] = []
}
