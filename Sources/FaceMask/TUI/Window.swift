import Foundation

public enum Window {
    public static var size: (rows: Int, cols: Int)? {
        var winsize = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &winsize) == 0 {
            return (Int(winsize.ws_row), Int(winsize.ws_col))
        } else {
            return nil
        }
    }
}
