import SwiftUI

// Design tokens from H3App-Design-Spec.md section 9. These are this
// redesign's own fixed values, not a claim that they match an OS-defined
// color exactly. Defined as a plain struct computed from colorScheme rather
// than an Asset Catalog, since this SPM executable target has none yet.
struct H3Palette {
    let canvas: Color
    let surface: Color
    let surfaceMuted: Color
    let stage: Color
    let textPrimary: Color
    let textSecondary: Color
    let border: Color
    let accent: Color
    let onAccent: Color
    let accentSoft: Color
    let errorColor: Color

    init(_ scheme: ColorScheme) {
        if scheme == .dark {
            canvas = Color(hex: 0x171B19)
            surface = Color(hex: 0x202623)
            surfaceMuted = Color(hex: 0x29312C)
            stage = Color(hex: 0x131814)
            textPrimary = Color(hex: 0xF0F4F0)
            textSecondary = Color(hex: 0xADB9B1)
            border = Color(hex: 0x404B43)
            accent = Color(hex: 0x8CDDBB)
            onAccent = Color(hex: 0x12231C)
            accentSoft = Color(hex: 0x2C4035)
            errorColor = Color(hex: 0xFFB4AB)
        } else {
            canvas = Color(hex: 0xF6F5F2)
            surface = Color(hex: 0xFFFFFF)
            surfaceMuted = Color(hex: 0xF0F2EE)
            stage = Color(hex: 0xE9EDE7)
            textPrimary = Color(hex: 0x202724)
            textSecondary = Color(hex: 0x606A64)
            border = Color(hex: 0xDCE2DB)
            accent = Color(hex: 0x176B57)
            onAccent = Color(hex: 0xFFFFFF)
            accentSoft = Color(hex: 0xE6F1EA)
            errorColor = Color(hex: 0xB42318)
        }
    }
}

private extension Color {
    init(hex: UInt32) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255
        )
    }
}

enum H3Spacing {
    static let xs: CGFloat = 4
    static let sm: CGFloat = 8
    static let md: CGFloat = 12
    static let lg: CGFloat = 16
    static let xl: CGFloat = 24
    static let xxl: CGFloat = 32
}

enum H3Radius {
    static let control: CGFloat = 8
    static let editor: CGFloat = 12
    static let stage: CGFloat = 16
}

enum H3ControlHeight {
    static let regular: CGFloat = 34
    static let primary: CGFloat = 44
}
