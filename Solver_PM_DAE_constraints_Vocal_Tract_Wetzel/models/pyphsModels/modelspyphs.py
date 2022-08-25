from cmath import e
import sympy as sp
import numpy as np
from pyphs import Core
from copy import copy


class PyphsBase():
    def __init__(self):
        self.core = Core('Empty')
        # States
        x = self.core.symbols('x')
        H = 0.5 * x**2
        self.core.add_storages(x, H)

        # Dissipative elements
        w = self.core.symbols('w')
        zw = w
        self.core.add_dissipations(w, zw)

        # Ports
        u, y = self.core.symbols(['u', 'y'])
        self.core.add_ports(u, y)

        # Interconnexion matrix
        self.core.init_M()
        Sxx, Sxw, Sxu, Swx, Sww, Swu, Syx, Syw, Syu = \
            self.core.symbols(['S_{xx}', 'S_{xw}', 'S_{xu}',
                               'S_{wx}', 'S_{ww}', 'S_{wu}',
                               'S_{yx}', 'S_{yw}', 'S_{yu}'])
        self.core.M = \
            sp.Matrix([[Sxx, Sxw, Sxu], [Swx, Sww, Swu], [Syx, Syw, Syu]])

    def compute_gradient(self):
        """Computes the symbolic expression of
        the hamiltonian gradient.

        Returns:
            array: hamiltonian gradient expression
        """
        return sp.derive_by_array(self.core.H, self.core.x)

    def compute_hessian(self):
        """Computes symbolic expression of the hessian of H
        and of the gradients of z(w, x)

        Returns:
            array: Hessian of H(x)
            array: Gradient of zw(w,x) wrt w
            array: Gradient of zw(w,x) wrt x
        """
        gradientH = self.compute_gradient()
        hessianH = sp.derive_by_array(gradientH, self.core.x)

        gradzw_w = sp.derive_by_array(self.core.z, self.core.w)
        gradzw_x = sp.derive_by_array(self.core.z, self.core.x)
        return hessianH, gradzw_w, gradzw_x

    def adim(self):
        """Function used to adimension the PHS.
        """
        # States
        self.xref = self.core.symbols(
            [f'{self.core.x[i]}_{{ref}}' for i in range(len(self.core.x))])
        Xref = sp.diag(*self.xref)
        xref_inv = [1/self.xref[i] for i in range(len(self.xref))]
        Xref_inv = sp.diag(*xref_inv)

        # Hamiltonian
        self.Href = self.core.symbols('H_{ref}')
        subs = dict(zip(self.core.x, [self.xref[i]*self.core.x[i] for i in range(len(self.core.x))]))
        self.core.H = 1 / self.Href * self.core.H.subs(subs)

        # Dissipation
        self.wref = self.core.symbols(
            [f'{self.core.w[i]}_{{ref}}' for i in range(len(self.core.w))])
        Wref = sp.diag(*self.wref)
        wref_inv = [1/self.wref[i] for i in range(len(self.wref))]
        Wref_inv = sp.diag(*wref_inv)

        self.zref = self.core.symbols(
            [f'z_{{{self.core.w[i]}ref}}' for i in range(len(self.core.z))])
        Zref = sp.diag(*self.zref)
        zref_inv = [1/self.zref[i] for i in range(len(self.zref))]
        Zref_inv = sp.diag(*zref_inv)

        for i in range(len(self.core.z)):
            subs = dict(zip(self.core.x, [self.xref[i]*self.core.x[i] for i in range(len(self.core.x))]))
            self.core.z[i].subs(subs)
            subs = dict(zip(self.core.w, [self.wref[i]*self.core.w[i] for i in range(len(self.core.w))]))
            self.core.z[i].subs(subs)
            self.core.z[i] = zref_inv[i] * self.core.z[i]

        self.uref = self.core.symbols(
            [f'{self.core.u[i]}_{{ref}}' for i in range(len(self.core.u))])
        Uref = sp.diag(*self.uref)

        self.yref = self.core.symbols(
            [f'{self.core.y[i]}_{{ref}}' for i in range(len(self.core.y))])
        Yref = sp.diag(*self.yref)
        yref_inv = [1/self.yref[i] for i in range(len(self.yref))]
        Yref_inv = sp.diag(*yref_inv)

        self.t_ref = self.core.symbols('t_{ref}')

        # Interconnexion matrix modification
        left = sp.diag(self.t_ref * Xref_inv, Wref_inv, Yref_inv)
        right = sp.diag(self.Href * Xref_inv, Zref, Uref)

        self.core.M = left * self.core.M * right

    def export_latex(self, filename, macro_name, macro):
        """Exports a text file with macros to generate
        model structure.

        Args:
            filename (string): filename
            macro_name (string): first part of macros names
            macro (string): optional macro to applyt to system elements
        """
        # State vector macro
        xCol = '#1'
        macro_states = f'\\newcommand\\{macro_name}states[1]' +\
                       '{' + '\\begin{equation}' +\
                       'x = ' + self.export_states(None, xCol, macro) +\
                       f'\\label{{eq:{macro_name}state}}' +\
                       '\\end{equation}' + '}'

        # Diss macro
        wCol = '#1'
        w, zw = self.export_diss(None, None, wCol, macro)
        macro_w = f'\\newcommand\\{macro_name}w[1]' +\
                  '{' + '\\begin{equation}' +\
                  'w = ' + w +\
                  f'\\label{{eq:{macro_name}w}}' +\
                  '\\end{equation}' + '}'
        zws = copy(self.core.z)
        for i, zw in enumerate(zws):
            for j, state in enumerate(self.core.x):
                colored_state = '\\textcolor{#2}{' + str(state) + '}'
                symbol = sp.symbols(colored_state)
                zw = zw.subs(state, symbol)
            for k, w in enumerate(self.core.w):
                colored_w = '\\textcolor{#1}{' + str(w) + '}'
                symbol = sp.symbols(colored_w)
                zw = zw.subs(w, symbol)
            zws[i] = sp.latex(zw)

        zw_text = '\\begin{bmatrix} '
        for zw in zws:
            zw_text += zw + '\\\\ '
        zw_text += '\\end{bmatrix}'
        macro_zw = f'\\newcommand\\{macro_name}zw[2]' +\
                   '{{' + '\\begin{equation}' +\
                   'z(\\textcolor{#1}{w}) = ' + zw_text +\
                   f'\\label{{eq:{macro_name}zw}}' +\
                   '\\end{equation}' + '}}'

        # Input/outputs macro
        uCol = '#1'
        u, y = self.export_io(None, None, uCol, macro)
        macro_u = f'\\newcommand\\{macro_name}u[1]' +\
                  '{{' + '\\begin{equation}' +\
                  'u =' + u +\
                  f'\\label{{eq:{macro_name}u}}' +\
                  '\\end{equation}' + '}}'
        macro_y = f'\\newcommand\\{macro_name}y[1]' +\
                  '{{' + '\\begin{equation}' +\
                  'y = ' + y +\
                  f'\\label{{eq:{macro_name}y}}' +\
                  '\\end{equation}' + '}}'

        # Structure macro
        xCol, wCol, uCol = '#1', '#2', '#3'
        macro_struct = f'\\newcommand\\{macro_name}struct[3]' +\
                       '{{' + '\\begin{equation}' +\
                       self.export_struct(None, xCol, wCol, uCol, macro) +\
                       f'\\label{{eq:{macro_name}struct}}' +\
                       '\\end{equation}' + '}}'

        # H(x) macro
        H = copy(self.core.H)
        for i, state in enumerate(self.core.x):
            if not macro:
                colored_state = '\\textcolor{#1}{' + str(state) + '}'
            else:
                colored_state =\
                    f'\\textcolor{{#1}}{{\\{macro}{{' + str(state) + '}}'
            symbol = sp.symbols(colored_state)
            H = H.subs(state, symbol)

        macro_H = f'\\newcommand\\{macro_name}H[1]' +\
                  '{{' + '\\begin{equation}' +\
                  'H(\\textcolor{#1}{x}) = ' +\
                  sp.latex(H) +\
                  f'\\label{{eq:{macro_name}H}}' +\
                  '\\end{equation}' + '}}'

        # Concatenation and export
        output = macro_states + '\n \n' + macro_w + '\n \n' + \
            macro_zw + '\n \n' + macro_u + '\n \n' + \
            macro_y + '\n \n' + macro_struct + '\n \n' + \
            macro_H

        with open(filename, 'w') as f:
            f.write(output)

    def export_states(self, filename, xCol, macro):
        """Exports a text file with latex state vector.

        Args:
            xCol (string): state color
            macro (string): optional macro to apply to x

        Returns:
            string: exported string
        """
        model = self.core
        states = '\\begin{bmatrix}'
        for state in model.x:
            if not macro:
                states += f'\\textcolor{{{xCol}}}{{{state}}}\\\\ '
            else:
                states +=\
                    f'\\textcolor{{{xCol}}}{{\\{macro}{{{state}}}}}\\\\ '
        states += '\\end{bmatrix}'
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(states)
        return states

    def export_diss(self, filename_w, filename_zw, wCol, macro):
        """Exports two text files with latex vectors of
        dissipation, elements and dissipation laws.

        Args:
            filename_w (string): filename for dissipative elements
            filename_zw (string): filename for dissipative laws
            wCol (string): dissipative elements color
            macro (string): optional macro to apply to w and z

        Returns:
            string: exported string for w
            string: exported string for zw
        """
        model = self.core
        ws = '\\begin{bmatrix}'
        for w in model.w:
            if not macro:
                ws += f'\\textcolor{{{wCol}}}{{{w}}}\\\\ '
            else:
                ws += f'\\textcolor{{{wCol}}}{{\\{macro}{{{w}}}}}\\\\ '
        ws += '\\end{bmatrix}'

        zws = '\\begin{bmatrix}'
        for zw in model.z:
            if not macro:
                zws += f'\\textcolor{{{wCol}}}{{{sp.latex(zw)}}}\\\\ '
            else:
                zws +=\
                    f'\\textcolor{{{wCol}}}\
                     {{\\{macro}{{{sp.latex(zw)}}}}}\\\\ '
        zws += '\\end{bmatrix}'

        if filename_w is not None:
            with open(filename_w, 'w') as f:
                f.write(ws)

        if filename_zw is not None:
            with open(filename_zw, 'w') as f:
                f.write(zws)
        return ws, zws

    def export_io(self, filename_u, filename_y,  uCol, macro=False):
        """Exports two text files with latex vectors of
        dissipation, elements and dissipation laws.

        Args:
            filename_u (string): filename for input
            filename_y (string): filename for output
            uCol (string): input/output color
            macro (string): optional macro to apply to y and u

        Returns:
            string: exported string for u
            string: exported string for y
        """
        model = self.core
        us = '\\begin{bmatrix}'
        for u in model.u:
            if not macro:
                us += f'\\textcolor{{{uCol}}}{{{u}}}\\\\ '
            else:
                us += f'\\textcolor{{{uCol}}}{{\\{macro}{{{u}}}}}\\\\ '
        us += '\\end{bmatrix}'

        ys = '\\begin{bmatrix}'
        for y in model.y:
            if not macro:
                ys += f'\\textcolor{{{uCol}}}{{{y}}}\\\\ '
            else:
                ys += f'\\textcolor{{{uCol}}}{{\\{macro}{{{y}}}}}\\\\ '
        ys += '\\end{bmatrix}'

        if filename_u is not None:
            with open(filename_u, 'w') as f:
                f.write(us)

        if filename_y is not None:
            with open(filename_y, 'w') as f:
                f.write(ys)
        return us, ys

    def export_struct(self, filename, xCol, wCol, uCol, macro=False):
        """Exports a text file with latex description of the model.
        The text needs to be put in an equation environment.

        Args:
            filename (string): filename
            xCol (string): state color
            wCol (string): dissipative elements color
            uCol (string): input/output color
            macro (string): optional macro name. Will be applied to all
                states, w, u, y, H and z

        Returns:
            string: exported string
        """
        model = self.core
        # We first build the left side of the equation
        left_side = '\\begin{bmatrix} '
        for state in model.x:
            if not macro:
                left_side += f'\\textcolor{{{xCol}}}{{\\dot{{{state}}}}}\\\\ '
            else:
                left_side += \
                    f'\\textcolor{{{xCol}}}\
                    {{\\dot{{\\{macro}{{{state}}}}}}}\\\\ '
        for w in model.w:
            if not macro:
                left_side += f'\\textcolor{{{wCol}}}{{{w}}}\\\\ '
            else:
                left_side += f'\\textcolor{{{wCol}}}{{\\{macro}{{{w}}}}}\\\\ '
        for i, y in enumerate(model.y):
            if i < len(model.y)-1:
                if not macro:
                    left_side += f'\\textcolor{{{uCol}}}{{{y}}}\\\\ '
                else:
                    left_side += f'\\textcolor{{{uCol}}}\
                    {{\\{macro}{{{y}}}}}\\\\ '
            else:
                if not macro:
                    left_side += f'\\textcolor{{{uCol}}}{{{y}}} '
                else:
                    left_side += f'\\textcolor{{{uCol}}}{{\\{macro}{{{y}}}}} '
        left_side += '\\end{bmatrix}'

        # Interconnexion matrix
        S = sp.latex(model.M)

        # Right side
        right_side = '\\begin{bmatrix} '
        for state in model.x:
            if not macro:
                right_side += \
                    f'\\nabla_{{{state}}} H(\\textcolor{{#1}}{{x}})\\\\ '
            else:
                right_side += \
                    f'\\nabla_{{\\tilde{{{state}}}}}\
                    \\{macro}{{H}}(\\textcolor{{#1}}{{\\tilde{{x}}}})\\\\ '
        zws = copy(model.z)
        for i, zw in enumerate(zws):
            for j, state in enumerate(model.x):
                if not macro:
                    colored_state = '\\textcolor{#1}{' + str(state) + '}'
                    symbol = sp.symbols(colored_state)
                    zw = zw.subs(state, symbol)
                else:
                    colored_state =\
                        f'\\textcolor{{#1}}{{\\{macro}{{' + str(state) + '}}'
                    symbol = sp.symbols(colored_state)
                    zw = zw.subs(state, symbol)
            for k, w in enumerate(model.w):
                if not macro:
                    colored_w = '\\textcolor{#2}{' + str(w) + '}'
                    symbol = sp.symbols(colored_w)
                    zw = zw.subs(w, symbol)
                else:
                    colored_w = \
                        f'\\textcolor{{#2}}{{\\{macro}{{' + str(w) + '}}'
                    symbol = sp.symbols(colored_w)
                    zw = zw.subs(w, symbol)
            zws[i] = sp.latex(zw)

        for i, z in enumerate(model.z):
            right_side += f'{zws[i]}\\\\ '
        for i, u in enumerate(model.u):
            if i < len(model.u)-1:
                if not macro:
                    right_side += f'\\textcolor{{{uCol}}}{{{u}}}\\\\ '
                else:
                    right_side +=\
                        f'\\textcolor{{{uCol}}}{{\\{macro}{{{u}}}}}\\\\ '
            else:
                if not macro:
                    right_side += f'\\textcolor{{{uCol}}}{{{u}}} '
                else:
                    right_side +=\
                        f'\\textcolor{{{uCol}}}{{\\{macro}{{{u}}}}} '
        right_side += '\\end{bmatrix}'

        full_text = left_side + '=' + S + right_side

        if filename is not None:
            with open(filename, 'w') as f:
                f.write(full_text)
        return full_text


class PyphsBaseTilde(PyphsBase):
    def __init__(self):
        self.core = Core('Empty')
        # States
        x = self.core.symbols('\\tilde{x}')
        H = 0.5 * x**2
        self.core.add_storages(x, H)

        # Dissipative elements
        w = self.core.symbols('\\tilde{w}')
        zw = w
        self.core.add_dissipations(w, zw)

        # Ports
        u, y = self.core.symbols(['\\tilde{u}', '\\tilde{y}'])
        self.core.add_ports(u, y)

        # Interconnexion matrix
        self.core.init_M()
        Sxx, Sxw, Sxu, Swx, Sww, Swu, Syx, Syw, Syu = self.core.symbols(
            ['\\tilde{S_{xx}}', '\\tilde{S_{xw}}', '\\tilde{S_{xu}}',
             '\\tilde{S_{wx}}', '\\tilde{S_{ww}}', '\\tilde{S_{wu}}',
             '\\tilde{S_{yx}}', '\\tilde{S_{yw}}', '\\tilde{S_{yu}}'])
        self.core.M = \
            sp.Matrix([[Sxx, Sxw, Sxu], [Swx, Sww, Swu], [Syx, Syw, Syu]])
