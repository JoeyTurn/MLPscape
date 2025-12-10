import heapq
from backend.utils import ensure_numpy, ensure_torch
import numpy as np
import torch

from itertools import permutations

# TODO: implement A and B initializations here I think and implement target initializations somewhere as well.
class Monomial(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __lt__(self, other):
        assert isinstance(other, Monomial)
        return self.degree() < other.degree()

    def degree(self):
        if len(self) == 0:
            return 0
        return sum(self.values())

    def max_degree(self):
        if len(self) == 0:
            return 0
        return max(self.values())

    def copy(self):
        return Monomial(super().copy())

    def __str__(self) -> str:
        if self.degree() == 0:
            return "1"
        monostr = ""
        for idx, exp in self.items():
            expstr = f"^{exp}" if exp > 1 else ""
            monostr += f"x_{{{idx}}}{expstr}"
        return f"${monostr}$"

    def __repr__(self):
        return self.__str__()

    def get_tensors(self, d: int) -> dict:
        tensors = {degree: torch.zeros([d] * degree) for degree in range(0, self.degree() + 1)}

        indices_power = []
        for index in self.keys():
            for _ in range(self[index]):
                indices_power.append(index)

        # Only unique permutations (account for duplicates where order doesn't matter)
        perms = list(set(permutations(indices_power)))

        for perm in perms:
            tensors[self.degree()][tuple(perm)] = 1 / len(perms)


        return tensors

    def evaluate(self, X, gamma=1.0):
        X = ensure_torch(X)
        
        N = X.shape[0]
        if self.degree() == 0:
            return torch.ones(N, device=X.device, dtype=X.dtype)

        y = torch.ones(N, device=X.device, dtype=X.dtype)
        for index, exponent in self.items():
            index = int(index)
            exponent = int(exponent)
            
            y.mul_((gamma * X[:, index]) ** exponent)

        return y

    @classmethod
    def from_repr(cls, s: str) -> "Monomial":
        """
        Parse strings like '$x_{0}^2x_{3}x_{10}^5$' or '$1$' into a Monomial.
        No regex used. Strict about format produced by __repr__/__str__.
        """
        if not isinstance(s, str):
            raise TypeError("from_repr expects a string")

        s = s.strip()
        if s.startswith("$") and s.endswith("$"):
            s = s[1:-1]
        s = s.replace(" ", "")

        if s in {"", "1"}:
            return cls()

        i, n = 0, len(s)
        out = {}

        def expect(ch: str):
            nonlocal i
            if i >= n or s[i] != ch:
                raise ValueError(f"Expected '{ch}' at pos {i} in {s!r}")
            i += 1

        def read_digits() -> int:
            nonlocal i
            start = i
            while i < n and s[i].isdigit():
                i += 1
            if start == i:
                raise ValueError(f"Expected digits at pos {start} in {s!r}")
            return int(s[start:i])

        while i < n:
            # x_{idx}
            expect('x')
            expect('_')
            expect('{')
            idx = read_digits()
            expect('}')

            # optional ^exp
            exp = 1
            if i < n and s[i] == '^':
                i += 1
                exp = read_digits()

            out[idx] = out.get(idx, 0) + exp

        return cls(out)
    
    def basis_factors(self, include_one: bool = False, canonical: bool = True):
        """
        Return a list of unit-degree Monomials whose product equals this monomial.
        Example: Monomial({0: 2, 3: 1}) -> [Monomial({0:1}), Monomial({0:1}), Monomial({3:1})]
        If degree == 0, returns [] unless include_one=True (then [Monomial({})]).
        If canonical=True, factors are ordered by increasing variable index.
        """
        if self.degree() == 0:
            return [Monomial({})] if include_one else []

        items = sorted(self.items()) if canonical else self.items()
        factors = []
        for idx, exp in items:
            for _ in range(int(exp)):
                factors.append(Monomial({idx: 1}))
        return factors
    
    def basis(self, canonical: bool = True) -> dict:
        if self.degree() == 0:
            return {}

        items = sorted(self.items()) if canonical else self.items()
        return {idx: int(exp) for idx, exp in items}


def compute_hea_eigval(data_eigvals, monomial, eval_level_coeff):
    hea_eigval = eval_level_coeff(monomial.degree())
    for i, exp in monomial.items():
        hea_eigval *= data_eigvals[i] ** exp
    return hea_eigval


def generate_hea_monomials(data_eigvals, num_monomials, eval_level_coeff, kmax=10, **kwargs):
    """
    Generates HEA eigenvalues and monomials in canonical learning order.

    Args:
        data_eigvals (iterable): data covariance eigenvalues
        num_monomials (int): Number of monomials to generate.
        eval_level_coeff (function): Function to evaluate kernel level coefficients.
        kmax (int): Search monomials up to degree kmax

    Returns:
        - hea_eigvals (np.ndarray): Array of HEA eigenvalues.
        - monomials (list): List of generated monomials.
    """
    try:
        num_monomials = abs(int(num_monomials))
    except Exception as e:
        raise ValueError(f"type(num_monomials) must be int, not {type(num_monomials)}") from e
    assert num_monomials >= 1
    data_eigvals = ensure_numpy(data_eigvals)
    d = len(data_eigvals)

    # populate priority queue with top monomial at each degree up to kmax
    pq = []
    pq_members = set()
    first_hea_eigval = compute_hea_eigval(data_eigvals, Monomial({}), eval_level_coeff)
    for k in range(1, kmax+1):
        monomial = Monomial({0: k})
        hea_eigval = compute_hea_eigval(data_eigvals, monomial, eval_level_coeff)
        # Each entry in the priority queue is (-hea_eigval, Monomial({idx:exp, ...}))
        pq.append((-hea_eigval, monomial))
        pq_members.add(repr(monomial))
    heapq.heapify(pq)
    
    monomials = [Monomial({})]
    hea_eigvals = [first_hea_eigval]
    for _ in range(num_monomials-1):
        if not pq:
            print("Warning: priority queue exhausted before reaching num_monomials.")
            return np.array(hea_eigvals), monomials
        neg_hea_eigval, monomial = heapq.heappop(pq)
        pq_members.remove(repr(monomial))
        hea_eigvals.append(-neg_hea_eigval)
        monomials.append(monomial)
        
        # generate successor monomials of same degree
        for idx in list(monomial.keys()):
            if idx + 1 < d:
                next_monomial = monomial.copy()
                next_monomial[idx] -= 1
                if next_monomial[idx] == 0:
                    del next_monomial[idx]
                next_monomial[idx + 1] = next_monomial.get(idx + 1, 0) + 1
                if repr(next_monomial) not in pq_members:
                    hea_eigval = compute_hea_eigval(data_eigvals, next_monomial, eval_level_coeff)
                    heapq.heappush(pq, (-hea_eigval, next_monomial))
                    pq_members.add(repr(next_monomial))

    return np.array(hea_eigvals), monomials


def get_monomial_targets(monomials, hea_eigvals, n_markers=20):
    target_monomials = [{0:1}, {10:1}, {100:1}, {190:1},
                        {0:2}, {0:1,1:1}, {1:1, 3:1}, {16:1,20:1}, {20:1,30:1},
                        {0:3}, {0:1, 1:1, 2:1}, {1:1, 3:1, 4:1}, {3:2, 5:1},
                        {0:4}]
    
    monomial_idxs = set()
    for tmon in target_monomials:
        if tmon not in monomials:
            print(f"Target {tmon} not in generated monomials. Skipping.")
            continue
        monomial_idxs.add(monomials.index(tmon))
    assert len(monomial_idxs) > 0

    # Add more modes: log-equidistant selection from hea_eigvals
    num_degrees = 3
    masked_eigvals = np.ma.masked_all((num_degrees, len(hea_eigvals)))
    for idx, monomial in enumerate(monomials):
        if 1 <= monomial.degree() <= num_degrees:
            masked_eigvals[monomial.degree()-1, idx] = hea_eigvals[idx]
    markers = np.logspace(np.log10(hea_eigvals[2_000]), np.log10(hea_eigvals[0]), n_markers)
    for i, marker in enumerate(markers):
        degree = i%num_degrees + 1
        idx = int(np.argmin(np.abs(masked_eigvals[degree-1] - marker)))
        if idx not in monomial_idxs:
            monomial_idxs.add(idx)
    return sorted(monomial_idxs)


def group_by_deg_max(monomials, stop_at_degree=None, assume_sorted=False):
    groups = {}
    for i, m in enumerate(monomials):
        deg = m.degree()
        if stop_at_degree is not None and deg > stop_at_degree:
            if assume_sorted:
                break
            continue
        key = (deg, m.max_degree())
        if key not in groups:
            groups[key] = []
        groups[key].append((i, m))
    return groups