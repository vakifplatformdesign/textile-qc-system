"""Delta E color difference calculations."""

import numpy as np

def deltaE76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 color difference formula."""
    d = lab1 - lab2
    return np.sqrt(np.sum(d**2, axis=-1))

def deltaE94(lab1: np.ndarray, lab2: np.ndarray, 
             kL: float = 1, kC: float = 1, kH: float = 1,
             K1: float = 0.045, K2: float = 0.015) -> np.ndarray:
    """CIE94 color difference formula."""
    L1,a1,b1 = lab1[...,0], lab1[...,1], lab1[...,2]
    L2,a2,b2 = lab2[...,0], lab2[...,1], lab2[...,2]
    dL = L1 - L2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C1 - C2
    da = a1 - a2
    db = b1 - b2
    dH_sq = da**2 + db**2 - dC**2
    dH_sq = np.maximum(dH_sq, 0)
    SL = 1
    SC = 1 + K1*C1
    SH = 1 + K2*C1
    dH = np.sqrt(dH_sq)
    return np.sqrt((dL/(kL*SL))**2 + (dC/(kC*SC))**2 + (dH/(kH*SH))**2)

def deltaE2000(lab1: np.ndarray, lab2: np.ndarray,
               kL: float = 1, kC: float = 1, kH: float = 1) -> np.ndarray:
    """CIEDE2000 color difference formula."""
    L1,a1,b1 = lab1[...,0], lab1[...,1], lab1[...,2]
    L2,a2,b2 = lab2[...,0], lab2[...,1], lab2[...,2]
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt((Cm**7) / (Cm**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = (np.degrees(np.arctan2(b1, a1p)) + 360) % 360
    h2p = (np.degrees(np.arctan2(b2, a2p)) + 360) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp)/2)
    Lpm = (L1 + L2)/2
    Cpm = (C1p + C2p)/2
    hp_sum = h1p + h2p
    hpm = np.where((np.abs(h1p - h2p) > 180), (hp_sum + 360)/2, hp_sum/2)
    T = 1 - 0.17*np.cos(np.radians(hpm - 30)) + 0.24*np.cos(np.radians(2*hpm)) + \
        0.32*np.cos(np.radians(3*hpm + 6)) - 0.20*np.cos(np.radians(4*hpm - 63))
    dRo = 30*np.exp(-((hpm - 275)/25)**2)
    Rc = 2*np.sqrt((Cpm**7) / (Cpm**7 + 25**7))
    Sl = 1 + (0.015*((Lpm - 50)**2)) / np.sqrt(20 + (Lpm - 50)**2)
    Sc = 1 + 0.045*Cpm
    Sh = 1 + 0.015*Cpm*T
    Rt = -np.sin(np.radians(2*dRo)) * Rc
    return np.sqrt((dLp/(kL*Sl))**2 + (dCp/(kC*Sc))**2 + (dHp/(kH*Sh))**2 + 
                   Rt*(dCp/(kC*Sc))*(dHp/(kH*Sh)))

def deltaE_CMC(lab1: np.ndarray, lab2: np.ndarray, 
               l: float = 2, c: float = 1) -> np.ndarray:
    """CMC l:c color difference formula."""
    L1, a1, b1 = lab1[...,0], lab1[...,1], lab1[...,2]
    L2, a2, b2 = lab2[...,0], lab2[...,1], lab2[...,2]
    
    dL = L1 - L2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C1 - C2
    da = a1 - a2
    db = b1 - b2
    dH_sq = da**2 + db**2 - dC**2
    dH_sq = np.maximum(dH_sq, 0)
    
    H1 = np.degrees(np.arctan2(b1, a1))
    H1 = np.where(H1 < 0, H1 + 360, H1)
    
    F = np.sqrt(C1**4 / (C1**4 + 1900))
    T = np.where((H1 >= 164) & (H1 <= 345),
                 0.56 + np.abs(0.2 * np.cos(np.radians(H1 + 168))),
                 0.36 + np.abs(0.4 * np.cos(np.radians(H1 + 35))))
    
    SL = np.where(L1 < 16, 0.511, (0.040975 * L1) / (1 + 0.01765 * L1))
    SC = ((0.0638 * C1) / (1 + 0.0131 * C1)) + 0.638
    SH = SC * (F * T + 1 - F)
    
    return np.sqrt((dL/(l*SL))**2 + (dC/(c*SC))**2 + (dH_sq/(SH**2)))

